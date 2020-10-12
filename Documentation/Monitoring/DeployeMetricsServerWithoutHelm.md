# **Deploye Metrics server without Helm**

check the node status

```bash
kubectl get nodes
```

download components.yaml file from [here](https://github.com/kubernetes-sigs/metrics-server/releases/download/v0.3.7/components.yaml) 

change the file name

```bash
mv components.yaml metrics-serv
```

check the node status.

```bash
kubectl top nodes
```

there is an error: metrics not available yet

deploy metric-server file

```bash
kubectl apply -f metrics-server.yaml
```

check the node and pods.

```bash
kubectl -n kube-system get all
kubectl top nodes
kubectl -n kube-system get pods
```

check the log for error .

```
kubectl -n kube-system logs -f metrics-server-!!!!!
```

edit the metrics-server file.

```
vi metrics-server.yaml
```

then find  deployment: metric-server section, container and  arges section and add these arguments below:

```
-     --kubelet-preferred-address-types=InternalIP

-     --kubelet-insecure-tls
```

 Save and exit.

then deploye matrics-server file again. 

```bash
Kubectl apply –f metrics-server.yaml
```

then run these below command and see the metrics.

```
kubectl top pods

kubectl top nodes
```

the metrics can be seen in dashboard too.





 

 