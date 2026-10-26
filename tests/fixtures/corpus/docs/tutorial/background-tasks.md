# Background Tasks { #background-tasks }

You can define background tasks to be run *after* returning a response.

## Using `BackgroundTasks` { #using-backgroundtasks }

Import `BackgroundTasks` and declare a parameter of that type, then call `background_tasks.add_task(write_notification, email)`.

## Caveat { #caveat }

If you need to perform heavy background computation across multiple servers, use bigger tools like Celery.
