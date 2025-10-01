In production, we usually use uv.lock to install frozen modules. However, in development we usually update our pyproject.toml. We sometimes forget to commit our updated uv.lock after we updated pyproject.toml.

```
torch.tensor(lst, device="cuda")
```
This is actually a blocking call. It first allocate a tenosr on GPU and then copy cpu data to GPU. The call on cpu side only proceed after the copy is finished.
```
torch.tensor(lst, device="cpu").to(device="cuda", non_blocking = True)
```
This is a proper async way to create a tensor with inital data.
```
torch.distributed.batch_isend_irecv(p2p_op_lst)
```
If all reference to tensors in the p2p_op_lst is vanshished, this could cause bug, since the tensor might be garbage collected while the async send hasn't finished.

Actual gpu applications are long. We usually only profile a short period of the lifespan of the application. The easiest way is to use `nsys launch`.
```
nsys launch myapp
```
In another shell
```
nsys start -o profile.nsys-rep
# after a short period of time
nsys stop
```
## Install Nsys
```
wget https://developer.nvidia.com/downloads/assets/tools/secure/nsight-systems/2025_2/NsightSystems-linux-cli-public-2025.2.1.130-3569061.deb
sudo apt install ./NsightSystems-linux-cli-public-2025.2.1.130-3569061.deb
```
