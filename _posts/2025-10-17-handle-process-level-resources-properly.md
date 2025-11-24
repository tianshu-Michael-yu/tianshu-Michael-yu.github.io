## Handle Process-Level Resources Properly

Some system resources are meant to exist only once per process — such as a GPU handle, a ZeroMQ (ZMQ) context, or a Torch distributed group. Mismanaging these can easily cause subtle bugs, especially in multiprocessing scenarios.

### The Problem

You often want all parts of your code to share the same ZMQ context. There are two common ways to do this:

**Approach 1: Pass the context through function arguments**
```python
create_worker_group(zmq_context, arg2, arg3)
create_worker(zmq_context, arg2, arg3)
```
This quickly becomes messy and leaks implementation details into functions that don’t even need to know about ZMQ.

**Approach 2: Use a global variable**
```python
zmq_context = zmq.Context()
def create_worker_group(...):
    global zmq_context
```
This works in a single process, but fails when you spawn new processes. The child process inherits the global variable, including the parent’s ZMQ context — which is invalid across processes. This bug is common in multiprocessing, RPC frameworks, or libraries like Ray.

### The Correct Solution

ZMQ provides a safer pattern using a per-process singleton:

```python
class Context:
    _instance = None
    _instance_lock = Lock()
    _instance_pid: int | None = None

    @classmethod
    def instance(cls, io_threads=1):
        if (
            cls._instance is None
            or cls._instance_pid != os.getpid()
            or cls._instance.closed
        ):
            with cls._instance_lock:
                if (
                    cls._instance is None
                    or cls._instance_pid != os.getpid()
                    or cls._instance.closed
                ):
                    cls._instance = cls(io_threads=io_threads)
                    cls._instance_pid = os.getpid()
        return cls._instance
```

Instead of storing the context manually, you always call:

```python
socket = zmq.Context.instance().socket(zmq.PUSH)
```

This ensures:
- Each process gets its own context.
- The API stays clean.
- You avoid passing globals across process boundaries.

### Takeaway

Never rely on global variables for process-level resources. Use per-process singletons (like `Context.instance()`) to ensure correctness and avoid hard-to-debug distributed errors.

### Other Bugs

Suppose you want each of your process has only one device visible to it. You launch an inference engine where process for rank 0 launches process for other rank using `torch.multiprocessing`. It turns out that you will almost always fail when you set `CUDA_VISIBLE_DEVICES` in the child process. Because `mp` implictly pickled the `torch` imported in rank 0 and send it to other rank, by the time you set the `CUDA_VISIBLE_DEVICES` it's too late. You already use the torch for `rank 0` unless you reimport `torch` again.