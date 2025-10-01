# tianshu-Michael-yu.github.io

In production, we usually use uv.lock to install frozen modules. However, in development we usually update our pyproject.toml. We sometimes forget to commit our updated uv.lock after we updated pyproject.toml.

```
torch.tensor(lst, device="cuda")
```
This is actually a blocking call. It first allocate a tenosr on GPU and then copy cpu data to GPU. The call on cpu side only proceed after the copy is finished.
```
torch.tensor(lst, device="cpu").to(device="cuda", non_blocking = True)
```
This is a proper async way to create a tensor with inital data.
