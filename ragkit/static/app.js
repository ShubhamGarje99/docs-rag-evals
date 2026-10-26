"use strict";
(() => {
  const $ = (s) => document.querySelector(s);
  const state = { config: null, retriever: null, turns: [], selected: null, busy: false };
  const SHORT = { vector: "Vector", bm25: "BM25", hybrid: "Hybrid", hybrid_rerank: "Hybrid + rerank" };

  // ---------- helpers ----------
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const store = {
    get(k) { try { return localStorage.getItem(k); } catch { return null; } },
    set(k, v) { try { localStorage.setItem(k, v); } catch { /* private mode */ } },
  };
  const fmtMs = (ms) => (ms == null ? "–" : ms >= 1000 ? `${(ms / 1000).toFixed(2)} s` : `${Math.round(ms)} ms`);
  const fmtCost = (c) => (c == null ? "–" : c < 0.01 ? `$${c.toFixed(5)}` : `$${c.toFixed(4)}`);

  function inline(text) {
    let s = esc(text);
    s = s.replace(/`([^`]+)`/g, (_, c) => `<code>${c}</code>`);
    s = s.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
    s = s.replace(/\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
    s = s.replace(/\[(\d{1,2})\]/g, '<button class="cite" data-n="$1" aria-label="Citation $1">$1</button>');
    return s;
  }

  // Headings/breadcrumbs keep Markdown emphasis and `code` from the docs; render them lightly.
  const titleHtml = (t) => esc(String(t || "").replace(/\*\*/g, "")).replace(/`([^`]+)`/g, "<code>$1</code>");

  function blocks(text) {
    return text.split(/\n{2,}/).map((b) => b.trim()).filter(Boolean).map((b) => {
      const lines = b.split("\n");
      if (lines.every((l) => /^\s*[-*] /.test(l))) {
        return `<ul>${lines.map((l) => `<li>${inline(l.replace(/^\s*[-*] /, ""))}</li>`).join("")}</ul>`;
      }
      if (lines.every((l) => /^\s*\d+[.)] /.test(l))) {
        return `<ol>${lines.map((l) => `<li>${inline(l.replace(/^\s*\d+[.)] /, ""))}</li>`).join("")}</ol>`;
      }
      if (/^#{1,6} /.test(b)) return `<p><strong>${inline(b.replace(/^#{1,6} /, ""))}</strong></p>`;
      return `<p>${lines.map(inline).join("<br>")}</p>`;
    }).join("");
  }

  function markdown(md) {
    return String(md || "").split("```").map((part, i) => {
      if (i % 2 === 0) return blocks(part);
      const nl = part.indexOf("\n");
      const code = nl >= 0 ? part.slice(nl + 1) : part;
      return `<pre><code>${esc(code.replace(/\n$/, ""))}</code></pre>`;
    }).join("");
  }

  // ---------- theme ----------
  function applyTheme(t) {
    if (t) document.documentElement.setAttribute("data-theme", t);
    else document.documentElement.removeAttribute("data-theme");
  }
  applyTheme(store.get("theme"));
  $("#theme-btn").addEventListener("click", () => {
    const cur = document.documentElement.getAttribute("data-theme")
      || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
    const next = cur === "dark" ? "light" : "dark";
    applyTheme(next);
    store.set("theme", next);
  });

  // ---------- config ----------
  async function loadConfig() {
    try {
      const r = await fetch("/api/config");
      if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
      const cfg = await r.json();
      state.config = cfg;
      const saved = store.get("retriever");
      state.retriever = cfg.retrievers.some((x) => x.id === saved) ? saved : cfg.default_retriever;
      renderRetrievers();
      const pill = $("#mode-pill");
      if (cfg.llm_enabled) {
        pill.textContent = `Claude · ${cfg.claude_model}`;
        pill.className = "pill on";
      } else {
        pill.textContent = "Retrieval-only";
        pill.className = "pill off";
        const b = $("#banner");
        b.textContent = `Retrieval-only mode: ${cfg.banner}`;
        b.hidden = false;
      }
      pill.title = `${cfg.chunks} chunks · ${cfg.embed_model}${cfg.rerank_model ? " · " + cfg.rerank_model : ""}`;
    } catch (e) {
      const b = $("#banner");
      b.textContent = `Backend not ready: ${e.message}`;
      b.hidden = false;
    }
  }

  function renderRetrievers() {
    const el = $("#retrievers");
    el.innerHTML = "";
    for (const r of state.config.retrievers) {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.setAttribute("role", "radio");
      btn.setAttribute("aria-checked", String(r.id === state.retriever));
      btn.title = r.label;
      btn.textContent = SHORT[r.id] || r.id;
      btn.addEventListener("click", () => {
        state.retriever = r.id;
        store.set("retriever", r.id);
        renderRetrievers();
      });
      el.appendChild(btn);
    }
  }

  // ---------- chat ----------
  function addUser(text) {
    $("#empty")?.remove();
    const div = document.createElement("div");
    div.className = "msg user";
    div.innerHTML = `<div class="bubble">${esc(text)}</div>`;
    $("#messages").appendChild(div);
    return div;
  }

  function addPending() {
    const div = document.createElement("div");
    div.className = "msg bot";
    div.innerHTML = `<div class="bubble"><span class="typing"><i></i><i></i><i></i></span></div>`;
    $("#messages").appendChild(div);
    div.scrollIntoView({ behavior: "smooth", block: "end" });
    return div;
  }

  function metaLine(d) {
    const parts = [`<span class="tag">${esc(SHORT[d.retriever] || d.retriever)}</span>`];
    parts.push(`<span>total <b>${fmtMs(d.latency_ms)}</b></span>`);
    parts.push(`<span>retrieval ${fmtMs(d.retrieval_latency_ms)}</span>`);
    if (d.llm_latency_ms) parts.push(`<span>Claude ${fmtMs(d.llm_latency_ms)}</span>`);
    if (d.usage) {
      parts.push(`<span>${d.usage.input_tokens.toLocaleString()} in / ${d.usage.output_tokens.toLocaleString()} out tok</span>`);
      parts.push(`<span>cost <b>${fmtCost(d.usage.cost_usd)}</b></span>`);
    }
    if (d.fallback_used) parts.push(`<span class="tag warn">served by fallback model</span>`);
    if (d.refused) parts.push(`<span class="tag warn">refused · ${esc((d.refusal_reason || "").replace(/_/g, " "))}</span>`);
    parts.push(`<span title="max cosine(question, retrieved chunks) vs MIN_RELEVANCE">relevance ${d.max_relevance?.toFixed(2)} / ${d.relevance_threshold}</span>`);
    return `<div class="meta">${parts.join("")}</div>`;
  }

  function renderBot(div, d) {
    const turn = { data: d, el: div };
    state.turns.push(turn);
    div.classList.toggle("refused", !!d.refused);
    let body;
    if (d.mode === "retrieval_only") {
      const top = d.sources.slice(0, 3).map((s) =>
        `<li><a href="${esc(s.url)}" target="_blank" rel="noopener">${titleHtml(s.title)}</a></li>`).join("");
      const gate = d.would_refuse
        ? "The refusal gate <strong>would refuse</strong> this question (low retrieval relevance)."
        : "The refusal gate would pass this question to Claude.";
      body = `<p><strong>Retrieval-only mode</strong> — no ANTHROPIC_API_KEY, so here are the top passages instead of an answer.</p><ol>${top}</ol><p class="muted">${gate}</p>`;
    } else {
      body = markdown(d.answer);
    }
    div.innerHTML = `<div class="bubble">${body}</div>${metaLine(d)}`;
    div.addEventListener("click", (e) => {
      const cite = e.target.closest(".cite");
      if (cite) { e.stopPropagation(); showCitation(turn, Number(cite.dataset.n), cite); return; }
      select(turn);
    });
    select(turn);
    div.scrollIntoView({ behavior: "smooth", block: "end" });
  }

  function renderError(div, msg) {
    div.classList.add("refused");
    div.innerHTML = `<div class="bubble"><p><strong>Error:</strong> ${esc(msg)}</p></div>`;
  }

  async function ask(question) {
    if (state.busy || !question.trim()) return;
    state.busy = true;
    $("#send").disabled = true;
    addUser(question);
    const pending = addPending();
    try {
      const r = await fetch("/api/chat", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ question, retriever: state.retriever }),
      });
      const d = await r.json();
      if (!r.ok) throw new Error(typeof d.detail === "string" ? d.detail : JSON.stringify(d.detail));
      renderBot(pending, d);
    } catch (e) {
      renderError(pending, e.message);
    } finally {
      state.busy = false;
      $("#send").disabled = false;
      $("#question").focus();
    }
  }

  // ---------- sources panel ----------
  function select(turn) {
    state.selected = turn;
    document.querySelectorAll(".msg.bot").forEach((m) => m.classList.toggle("selected", m === turn.el));
    const d = turn.data;
    const citedN = {};
    (d.citations || []).forEach((c) => { citedN[c.chunk_id] = c.n; });
    $("#sources-meta").textContent = `${SHORT[d.retriever] || d.retriever} · top-${d.sources.length} · ${fmtMs(d.retrieval_latency_ms)}`;
    const list = $("#source-list");
    list.innerHTML = "";
    d.sources.forEach((s) => {
      const li = document.createElement("li");
      li.className = "src" + (citedN[s.chunk_id] ? " cited" : "");
      li.dataset.chunk = s.chunk_id;
      const scores = Object.entries(s.detail || {}).map(([k, v]) =>
        `<span class="tag">${esc(k.replace(/_/g, " "))} ${Number.isInteger(v) ? v : Number(v).toFixed(3)}</span>`).join("");
      li.innerHTML = `
        <div class="src-top"><span class="rank">#${s.rank}</span>
          <a class="src-title" href="${esc(s.url)}" target="_blank" rel="noopener">${titleHtml(s.title)}</a>
          ${citedN[s.chunk_id] ? `<button class="cite" data-n="${citedN[s.chunk_id]}">${citedN[s.chunk_id]}</button>` : ""}</div>
        <div class="src-id">${esc(s.chunk_id)}</div>
        <div class="src-scores">${s.relevance != null ? `<span class="tag">cosine ${s.relevance.toFixed(3)}</span>` : ""}${scores}</div>
        <div class="src-text">${esc(s.text)}</div>
        <button class="more" type="button">Show more</button>`;
      li.querySelector(".more").addEventListener("click", (e) => {
        const t = li.querySelector(".src-text");
        t.classList.toggle("open");
        e.target.textContent = t.classList.contains("open") ? "Show less" : "Show more";
      });
      li.querySelector(".src-top .cite")?.addEventListener("click", (e) => showCitation(turn, citedN[s.chunk_id], e.target));
      list.appendChild(li);
    });
  }

  function showCitation(turn, n, anchor) {
    if (state.selected !== turn) select(turn);
    const c = (turn.data.citations || []).find((x) => x.n === n);
    if (!c) return;
    const pop = $("#popover");
    const quotes = (c.quotes && c.quotes.length ? c.quotes : [c.quote]).map((q) => `<blockquote>${esc(q)}</blockquote>`).join("");
    pop.innerHTML = `<div class="pt">[${n}] ${titleHtml(c.title)}</div>${quotes}<a href="${esc(c.url)}" target="_blank" rel="noopener">Open in the FastAPI docs ↗</a>`;
    pop.hidden = false;
    const r = anchor.getBoundingClientRect();
    const w = Math.min(380, window.innerWidth - 24);
    pop.style.left = `${Math.max(12, Math.min(r.left, window.innerWidth - w - 12))}px`;
    const below = r.bottom + 8;
    pop.style.top = `${below + pop.offsetHeight > window.innerHeight ? Math.max(12, r.top - pop.offsetHeight - 8) : below}px`;
    const src = document.querySelector(`.src[data-chunk="${CSS.escape(c.chunk_id)}"]`);
    if (src) { src.scrollIntoView({ behavior: "smooth", block: "nearest" }); src.classList.remove("flash"); void src.offsetWidth; src.classList.add("flash"); }
  }
  document.addEventListener("click", (e) => {
    if (!e.target.closest(".popover") && !e.target.closest(".cite")) $("#popover").hidden = true;
  });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") $("#popover").hidden = true; });

  // ---------- composer ----------
  const ta = $("#question");
  ta.addEventListener("input", () => { ta.style.height = "auto"; ta.style.height = `${Math.min(ta.scrollHeight, 160)}px`; });
  ta.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("#composer").requestSubmit(); }
  });
  $("#composer").addEventListener("submit", (e) => {
    e.preventDefault();
    const q = ta.value;
    ta.value = "";
    ta.style.height = "auto";
    ask(q);
  });
  $("#examples")?.addEventListener("click", (e) => {
    const b = e.target.closest("button[data-q]");
    if (b) ask(b.dataset.q);
  });

  loadConfig();
})();
