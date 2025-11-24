## Pipeline Parallelism in Inference

In the previous post, [Inference Engine]({% post_url 2025-11-22-inference-engine %}), we dived into how to develop an async inference engine.

If we throws pipeline parallelism into the whole thing, it gets complicated.

Let's show a naive implementation of infer.

```python
class Inferencer:
    def infer(self):
        ctx = self._get_ctx()
        output = self._model.forward(ctx)
        self._send_to_next_rank(output, ctx)
```
That mean in each cycle we always start with receiving tensors necessary for doing computation, do the computation and send the output to next rank. The code looks clean and easy to follow. But it produces the following result.

![PPDeadlock](/img/PPDeadlock.png)

As you can see the dependency arrow cross. This means if the send and recv communication are both blocking operation, there will be a dead lock. For sending GPU tensor, we can put send and recv on different stream, hence no blocking operation. But for sending cpu meta data, we need to use `torch.distributed.send_obj_list` which is a synchronized call.

The solution is basically you switch the order of send and recv on either the first pp rank or the last pp rank. In my experience, the first design just works better than the second.

![GoodPPTimeline](/img/GoodPPTimeline.png)

This is the first design. Basically ever other rank is exactly the naive case, except the first rank.
```python
# first rank cycle
def infer(self, batch, should_recv_out):
    ctx = self._prepare_ctx(batch)
    output = self._model.forward(ctx)
    if should_recv_out:
        token_id = self._recv_token_id()
    self._send_ctx(output, ctx)
    if should_recv_out:
        return token_id
```

This looks complicated, but all the complexity actually happens at the first rank. `should_recv_out` determines whether the first rank is going to recveive the `token_id` from the last pp rank in this cycle. This signal can be calculated and given by scheduler. This feels natural as the scheduler is really the one that's keeping track of the movement of batches in the system. 

![UglyPPTimeline](/img/UglyPPTimeline.png)

This is the second design. This time both the first rank and the last rank is different. But the first rank follows the patter of doing receiving tensor first, then computation, and then sending the tensor to the next rank. It's the last rank that will looks weird.

```python
def infer(self):
    ctx = self._get_ctx()
    if self._last_token_id:
        self._send_token_id(self._last_token_id)
    token_id = self._model.forward(ctx)
    self._last_token_id = token_id
```

This gives you a rough idea about the weirdness of this design. Suddenly, the inferencer has this weird state called `self._last_token_id` that it has to track across cycle. Remember for the easiness of the test, we always want the call to be stateless. If the behavior of the call changes because of the previous call, it's much harder test. You may also notice the above actually doesn't quite work. What if you don't have more incoming batch? Then, you just stuck on the `self._get_ctx()` call and never send the last `token_id` back. You can fix it by throwing more logic. For instance, you can clearly define cycle on every pp rank so that it matches the cycle on the first rank. The code just becomes ugly really fast. In no time, you find you are doing complex scheduling on the rank that you are not supposed to do any.

This is why the first design is supperior. You can think the first rank just doing all the scheduling and passed the work to other rank, the other ranks completed their work and send final result back. Because all the messy stateful logic is in the first rank, we can concentrate those logic in the scheduler. Then as long as we throughly tested the behavior of the scheduler, we are good. 

What should cycles be managed on different pp rank in code? Here's a sketch.

```
class Engine:   
    def first_rank_loop(self):
        while True:
            next_batch, requests_to_receive = self._scheduler.schedule()
            token_ids = self._inferencer.infer(next_batch, requests_to_receive)
            completed_requests = self._output_processor.process(requests_to_receive, token_ids)
            self._scheduler.cleanup_requests(completed_requests)

class Worker:
    def other_rank_loop(self):
        while True:
            self._inferencer.infer([])
```

This design makes the work on the other rank very simple. Now there's still many details to be filled. One interesting thing is how do you gracefully terminate all those loops? One simple idea is that we can leverage `torch.multiprocessing.Event`. We can have a shutdown event `shutdown_event = mp.Event()` has this as the loop termination condition. If we set the event, all loop will be terminated. But it will not happen. Because by the end, when you don't have any new batches coming in, you loop will be blocked on the waiting to receive new ctx call. You will never move to the while condition. 

One solution is that you can send a special ctx that tells you that you should shutdown and then the inferencer pass that ctx to the next rank. It's kind of annoying that the return type of the `infer` call can also be a shutdown signal. Another way is that you pass the `next_batch` to the next rank through `mp.Queue`. So in each pp rank we need to do `prepare_ctx` instead of doing `get_ctx` for all other pp rank. We kind of doing some wasteful work every rank but we also simplify the `infer` code quite a bit. As now every rank has similar behaivor now. An added benefit is that `torch.multiprocessing` enable you to pass CPU tensor through shared memory, so you can get those meta data pretty quickly. However, as PP is mostly advantagous in the multinode setting, `torch.multiprocessing` is not going to work. We still have to use `torch.distributed` like the first method. So for all the complexity it involves I think the first solution is still better. 
