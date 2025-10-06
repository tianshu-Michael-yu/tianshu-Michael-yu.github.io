## How to solve performance issues

Let's start with a profile of the inference engine. 

![Alt text](/img/dp_2_tp_4_nsys_timeline_1.png)

This is the nsys profiling result of the rollout phase of a RL step. The inference configuration is the following. We are using a machine with 8 H100 GPU. On this machine, we have two inference engines running. Each inference engine instance uses tensor parallel size 4, i.e. TP = 4.

As you observed, the most obivous problem is that there're so many bubbles between each inference step. To understand the cause, we need to focus on one inference engine. But nsys didn't tell use which process belong to which inference engine. Luckily, we can reason using our knowledge about model parallelism. Since tensor parallelism do 2 all-reduce in each decode layers, the forward for each shard for that engine should end at approximately the same time. Let's pin those processses.

![Alt text](/img/dp_2_tp_4_nsys_timeline_2.png)

Let's take tp rank 3 as an example. The reason for the bubble is that the Cudagraph launch is delayed as we can see on the CPU side. But if we inspect the CPU timely more carefully by clicking on the `Threads` dropdown for each process, we understand that the delay of cudagraph is caused by delay of gen_context preparation and that is caused by the abnormally long zmq_bcast_ctx_meta.

![Alt text](/img/dp_2_tp_4_nsys_timeline_3.png)

Here there're two natural hypothesis to explain that.
1. The interprocess communication is too slow.
2. The serialization of python objects take too long. 

Usually, we have to test each of the hypothesis. However, as a good engineer, my intuition tells me that 2 is more probable. Since the interprocess communication here uses ipc directly instead of network, bandwith shouldn't be a problem. For the second hypothesis to be true, it might be that the python objects in gen_ctx_meta are too large to be serialized. We just need inspect what's inside gen_ctx_meta to verify our hypothesis. Indeed, it turns out that in our inference engine, the `slots_mappings` are stored in gen_ctx_meta and its size is proportional to `batch_size*seq_lens`. If it takes the sender 3ms to serialize the python object, it will take another 3ms to deserialize the python object on the receiver end. In long sequence scenario, it's simply too large so that the non scheduler rank's cudagraph launch will be delayed significantly.

![Alt text](/img/dp_2_tp_4_nsys_timeline_4.png)

As you can see, after we remove `slots_mappings` from the data to be sent, the cudagraph for all tp rank starts at approximately the same time and there're no bubbles on the timeline.

