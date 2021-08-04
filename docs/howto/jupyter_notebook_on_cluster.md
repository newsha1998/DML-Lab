# Jupyter Notebook on K8s cluster

In order to deploy a jupyter notebook server on K8s cluster, kubeflow can be employed. For this purpose, a `yaml` file should be configured and applied on the cluster. The `apiVersion` and `kind` should be set like below

```{yaml}
apiVersion: kubeflow.org/v1
kind: Notebook
```

This configures the new resource to be a Notebook within kubeflow. The rest of the file consists of `spec` of the resource which in this case contain

- `serviceAccount`
- `volumes`
- `containers`

An example of a complete `yaml` file for setting up a jupyter notebook can be found [here](https://github.com/kubeflow/kubeflow/blob/master/components/notebook-controller/loadtest/jupyter_test.yaml).

Using the `notebook.yaml` file, the notebook can be created by `kubectl` command`

```
kubectl create -f notebook.yaml
```

This command will create a pod which has the jupyter notebook running on and a service for that pod. Using the `ip` and `port` of the service, the notebook can be accessed.