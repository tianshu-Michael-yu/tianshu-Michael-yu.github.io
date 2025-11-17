## Inference Engine

This post will introduce the core components of an inference engine.

![InferenceEngineStructrue](/img/inference_engine_structure.png)

Engine basically orchestrate those workers to complete inference task. Each worker is an independent process that manage one device (e.g. a GPU). Within each worker process, the inferencer does essentially all the GPU work, the model builder essentially load the model shard that will be run on that device. The scheduler defines what goes into each batch and the block usage. The block manager helps the scheduler to mange blocks. This is for paged attention. 

You may ask the question why don't we simply create out model within the inferencer. That's majorly for the easy of unittest. In reality, the logic for creating the model object and loading model shard efficently is very complicated. But once you have the model shard you don't usually change it much. By separating out the model creation logic, we simplify the behavior of inferencer to simply manage the computation on GPU. We don't need to pass in a complicated config. 

```python3
class Engine:
    def add_requests(self, requests: List[Request]):
        ...

    def _step(self):
        batch = self._scheduler.schedule()
        self._workers.execute("queue_batch", batch)
    
    def generate(self):
        while True:
            self._step()

    def _update_output(self)
        ...
```

The `add_requests` interface allows user to add more request to the engine asynchronously without blocking main operation cycle. It's usually achieved through adding to a waiting queue, while scheduler fetch from this queue to schedule the next batch. The `self._workers.execute` is for invoking a function (e.g. `queue_batch`) in all of those worker process. In practice, it's usually done through rpc call. The `_update_output` is a remote function that the worker can call, when it emit a token or other form of output, so that the user can see the result.

```python3
class Worker:
    def queue_batch(self, batch):
        ...
    
    def worker_loop(self):
        while True:
            batch = self._next_batch()
            output = inferencer.infer(batch)
            self._output_processor.process(output)
```

The worker gets more batch as the engine sends batch to it through the remote call `queue_batch`. The worker loop is very simple, get the next_batch in the queue, call the inferencer to do the inference and finally let the output processor to process the output. The output processor don't do any real work on GPU. It simply wait for the result of the inference from transfering from GPU to CPU and process it. The processing may also include send the result back to the engine.


```python3
class Inferencer:
    def infer(self, batch):
        ctx = self._prepare_context(batch)
        output = self._forward(ctx)
        return self._sampler.sample(output)
```

The inferencer manages all the real compute on GPU. It start by preparing context. It's basically a way to pack all the necessary information such as input id, position, attention mask, etc to different tensors and allocate those tensors on GPU. Then `self._forward` do the computation. Finally, if needed, we let the sampler to sample the result token. 

## Sync to Async

The most naive implementation of the workflow is like the following.

![NaiveTimeline](/img/naive_inference_timeline.png)

* prep_ctx: Create context for forward. And launch the allocation of those tensor on GPU.
* forward: On CPU, it's basically launch the computation on GPU. On GPU, it's basically doing the computation
* proc_out: process the output of GPU computation. This mean we have to wait for the result on GPU.

One characters of working with GPU is that it allows you to dispatch the GPU work asynchronously. i.e. The CPU doesn't have to wait for the result of GPU unless it's necessary. This allow us to launch many GPU workload to keep GPU busy. The problem is clearly that in this work flow you cannot schedule the next batch until the current batch is completed, sent backed to the CPU, and processed. This creates this bubble where the GPU has to wait for CPU for more work. For a workload like serving qwen3_30b, on 1 H100, the bubble could accounts for a 20% loss in per token latency and throughput.

The solution for this problem is to do async scheduling.

![AsyncTimeline](/img/async_inference_timeline.png)

Originally proc_out of cycle i will just process the GPU output of cycle i. Now, instead, it process the output from its previous cycle i-1. This way the CPU overhead of postprocessing and scheduling next batch is hidden by the GPU compute.  

How can the above code structure achieve this async way of inferencing? We can implement our output processor differently.

```
class OutputProcessor:
    def __init__(self):
        self._prev_output = None

    def process(self, output):
        if self._prev_output:
            self._prev_output.synchronize()
            self._process(prev_output)
        self._prev_output = output
```

So instead of processing the current output immediately, we process the previous output and then update the prev_output as the current output.

There are many other things need to be changed as well. For instance,  when we prepare the context for cycle i, we don't necessessarily have the result of cycle i-1 ready yet. We need to reserve a space in the input_sequence for cyle i for the result of cycle i-1. This requires the modification of scheduler and prepare_ctx. Another modification is the termination logic. Since in cycle i, we no longer processing the output of cycle i, we don't know if cycle i generated the final token. We will only figure out that in the next cycle. So we always going to do an extra cycle after the termination token. 