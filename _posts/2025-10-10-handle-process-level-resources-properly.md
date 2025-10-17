## Handle Process Level Resources Properly

There're many resources should be a single process. For instance, a process should manage only one GPU, a process should only have one zmq context,
and a process should only have one torch distributed group. It's a very common pattern.

Let's take an example. We want all zmq sockets in a process use the same zmq context.

There are two approaches:
1. Pass the context around through function signature or `__init__` method explictly.
2. Store the context as a global variable.

The problem for approach 1 is that your function will be littered with

```
create_worker_group(zmq_context, arg2, arg3, ...)
create_worker(zmq_context, arg2, arg3, ...)
```

Even though those functions conceptually doesn't handle zmq or they don't even have to eventually use zmq underneath (imagine zmq is just one communication
backend), they have to pass this context through. In reality, we may have many system resources similar to zmq context. 