# CORS (Cross-Origin Resource Sharing) { #cors-cross-origin-resource-sharing }

CORS refers to the situations when a frontend running in a browser has JavaScript code that communicates with a backend in a different origin.

## Use `CORSMiddleware` { #use-corsmiddleware }

You can configure it in your **FastAPI** application using the `CORSMiddleware`.

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:8080"])
```

The `max_age` argument sets a maximum time in seconds for browsers to cache CORS responses. Defaults to `600`.
