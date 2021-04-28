# Distributed Data-Parallel Training with Kubeflow/Pytorch

To train a distributed model using pytorch and kubeflow you have to take the following steps:
1. Write your code
2. Create a Dockerfile for your code and build it to a docker image
3. Create a manifest file 
4. Submit your job into the cluster

## Distributed Data-Parallel Pytorch Training
In data-parallel paradigm model is copied into each gpu (cpu) device and fed with a portion of data. Pytorch takes care of synchronizing gradients between processes. Processes can be on the same node (device), or on different nodes. In Kubeflow, since every process runs on its own pod, each pod can be thought of as a node with an ip address and a port.
The main differences between a distribited data-parallel training and usual single gpu training is as follows:
1. Initializing a process group on each process. Process group takes care of communication between nodes. There are several backends that can be used. Gloo is recommended when training on cpu and NCCL is recommended for gpu training. To initialize a process group you have to specify master node's ip and port (master node is the node with rank 0). This is done by kubeflow using environment variables. (more info: [torch.distributed](https://pytorch.org/docs/master/distributed.html))
```python
torch.distributed.init_process_group(torch.distributed.Backend.GLOO)
```
2. Wraping model in Distributed Data-Parallel Module: For gpu models
```python
model = torch.nn.parallel.DistributedDataParallel(model)
```
and for cpu models
```python
model = torch.nn.parallel.DistributedDataParallelCPU(model)
```
thats it (more info: [torch.nn.parallel.DistributedDataParallel](https://pytorch.org/docs/master/generated/torch.nn.parallel.DistributedDataParallel.html)).

## Creating Dockerfile and Docker Image
There are base images for every version of pytorch on [docker hub](https://hub.docker.com/r/pytorch/pytorch/tags). Using the appropariate version of pytorch as base image you have to install dependencied using pip and move your code and data into the image (data can be downloaded e.g. torchvision.datasets).
(more info about writing a Dockerfile: [docker documentation](https://docs.docker.com/compose/gettingstarted/#step-2-create-a-dockerfile)) (example [Dockerfile](Dockerfile))

To build your docker image you have to specify a tag using the following command:
```bash
docker build -t [tag] ./
```
## Creating Manifest File
Manifest file is used to submit your job into cluster. You have to specify a docker image and required resources for each Master/Worker pod and pass in the parameters to your code. You can change replica to create mode workers. (See examples: [manifest-cpu.yaml](manifest-cpu.yaml), [manifest-gpu.yaml](manifest-gpu.yaml))

## Submit
```bash
kubectl create -f [manifest file]
```

To see the logs:
```bash
kubectl logs -f [pod name]
```
