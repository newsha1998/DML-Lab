# Using Grafana to monitor jobs on k8s

**Note**: There is also a video for this tutorial [here](https://drive.google.com/file/d/1h3GNt4o9ZfPY8_VI7Y-bHecK_IvfKi7B/view?usp=sharing)(Persian)
## List of contents
1. what are Grafana and Prometheus
2. How to connect to Grafana dashboard
3. How to use the Grafana dashboard
4. Conclusion

## What are Grafana and Prometheus
* Prometheus is a datasource which works like a back-end. It collects various metrics for jobs running on a k8s cluster.
* Grafana is a tool to visualize metrics stored by Prometheus (or other datasources such as Influxdb)

## How to connect to Grafana dashboard
Grafana is currently running as a service on the k8s cluster. One can connect to its dashboard on local computer by following these steps:
1. Forward Grafana service port on the remote server port (if not already).
This can be done by the following line (note that on our cluster: ```[grafana-service]=spark-dashboard-grafana```)
```
kubectl port-forward service/[grafana-service] 3000:3000
```
2. Forward the port on which Grafana is serving on to your local computer port (which is port 3000). This can be done by: (note that on our cluster: ```[remote server]=drrohban@192.168.207.154```)
``` 
ssh -L 3000:localhost:3000 [remote server]
``` 

On Windows this can also be done in MobaXterm tunneling option.

3. Go to ```localhost:3000``` on your own browser where you should see the login page of Grafana.
4. Login by entering admin for both username and password.

## How to use the Grafana dashboard
Once you have succesfully loged in to Grafana, you should see all the current dashboards. The ***K8 Cluster Detail Dashboard*** can be used to monitor metrics such as CPU usage, memory usage, and so on for the cluster as a whole. The ***Analysis by Pod*** is a dashboard to monitor metrics of pods individually. Choose the namespace and the pod on the top menu of the dashboard to see the different metrics of that particular pod. Be careful to choose the correct **time range** on the top right corner for your usage.

## Conclusion
We saw how to use Grafana to monitor a k8s cluster and individual pods running on the cluster. There are many more dashboards for Grafana on the grafana dashboards [website](https://grafana.com/grafana/dashboards)
