# Install Nvidia driver

`sudo apt install nvidia-driver-450`
After installation, the system needs a reboot.

# Install Docker

Update apt package:
`sudo apt-get update`
Install docker dependencies:
``` 
sudo apt-get install \
apt-transport-https \  
ca-certificates \
curl \  
gnupg-agent \ 
software-properties-common
```
Add docker repository
```
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -

sudo add-apt-repository \  
    "deb [arch=amd64] https://download.docker.com/linux/ubuntu \  
    $(lsb_release -cs) \  
    stable"
```
Update package database again:
`sudo apt-get update`
    
 Install docker:
`sudo apt-get install docker-ce  docker-ce-cli containerd.io`


# Install Nvidia container toolkit
In order for the K8s to identify gpus, the nvidia toolkit must be installed. First add the toolkit gpgkey:
```
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \  
&& curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \  
&& curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
```
Install nvidia toolkit:
```
sudo apt-get update 
sudo apt-get install -y nvidia-docker2
sudo  systemctl restart docker
```

Add the nvidia container runtime to docker daemon. Change the docker daemon file `/etc/docker/daemon.json` and set the default runtime to nvidia:
```
{
    "default-runtime": "nvidia",
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    }
}
```

# Install kubeadm & kubectl & kubelet

Install these packages on all of the nodes:
```
sudo apt-get update && sudo apt-get install -y apt-transport-https curl
curl -s https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -

cat <<EOF | sudo tee /etc/apt/sources.list.d/kubernetes.list   
deb https://apt.kubernetes.io/ kubernetes-xenial main
EOF
    
sudo apt-get update
sudo apt-get install -y kubelet=1.14.10-00 kubeadm=1.14.10-00 kubectl=1.14.10-00
sudo apt-mark hold kubelet  kubeadm  kubectl
```
Disable swap:
`sudo swapoff -a`

## Initialize kubernetes cluster
Initialize kubeadm:
`sudo  kubeadm  init  --apiserver-advertise-address=<master-ip> --pod-network-cidr=10.244.0.0/16`

Install flannel (for networking):
`kubectl apply -f https://raw.githubusercontent.com/coreos/flannel/master/Documentation/kube-flannel.yml`

Untaint master node:
`kubectl taint nodes --all node-role.kubernetes.io/master-`

Install K8s dashboard:
`kubectl apply -f https://raw.githubusercontent.com/kubernetes/dashboard/v2.0.0/aio/deploy/recommended.yaml`

Install Nvidia device plugin:
`kubectl create -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.8.0/nvidia-device-plugin.yml`
