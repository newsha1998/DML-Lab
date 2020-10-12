### Kubernetes Monitoring

After you successfully setup kubernetes cluster, the next step is to configure a proper monitoring and alerting mechanism.

For monitoring we need these:

1. #####  InfluxDB: InfluxDB is a time series database optimized for high-availability storage and rapid retrieval of time series data.

2.  Grafana: Grafana can visualize data from multiple monitoring solutions. It presents nice dashboards and comes with built-in alerting. Plug in your favorite data source, and you’re ready to go.

3.  Prometheus: Prometheus collects metrics from monitored targets by scraping metrics HTTP endpoints on these targets.

 Download and extract the Yaml file from [here]( https://octoperf.com/img/blog/kraken-kubernetes-influxdb-grafana-telegraf/kraken-monitoring.zip):    

```bash
kubectl get configmap influxdb-config --export -o yaml >> influxdb-config.yaml
```

check that the ConfigMap is created

```bash
kubectl describe configmap influxdb-config
```

#### Map environment variables using secrets

A secret is an object that contain a small amount of sensitive data such as a password, a token, or a key. 

Start by creating the configuration file influxdb-secrets.yaml

```bash
kubectl apply -f influxdb-secrets.yaml
```

 Display the created *Secret*, environment variable values are not visible

```bash
kubectl describe secret influxdb-secrets
```

####  Mount a Data Volume

In kubernetes, persistence of data is done using Persistent volumes. A PersistentVolumeClaim describe the type and details of the volume required. Kubernetes finds a previously created volume that fits the claim or creates one with a dynamic volume provisioner.

let's try another tool to generate declarative configuration files for kubernetes: Kompose. Kompose takes a Docker Compose file and translates it into kubernetes resources.

Installation is straightforward:

```
curl -L https://github.com/kubernetes/kompose/releases/download/v1.18.0/kompose-linux-amd64 -o kompose 

chmod +x kompose

sudo mv ./kompose /usr/local/bin/kompose
```

  Create the following `docker-compose.yml` file

```
version: '3.0'  
  
services:  
  influxdb:  
    image: influxdb:1.7.4  
    container_name: influxdb  
    expose:  
      - "8086"  
    env_file:  
      - 'env.influxdb'  
    volumes:  
      - influxdb-data:/var/lib/influxdb  
      - ./influxdb.conf:/etc/influxdb/influxdb.conf:ro  
    restart: unless-stopped
```

Run the kimpose convert command to convert the created docker-compose.yaml file into several k8s configuration files:

```
kompose convert -f docker-compose.yml
```

 It generates a file named `influxdb-data-persistentvolumeclaim.yaml` though. Rename it to influxdb-data.yaml and update the storage capacity:

Finally create the PersistentVolumeClain:

```
kubectl apply -f influxdb-data.yaml
```

 And check that the created PVC is matched to a PersistentVolume:

```
kubectl get pvc influxdb-data
```

 you can see the volume name pvc-xxx and Bound status there.

#### Create an InfluxDB Deployment

Let’s apply an InfluxDB Deployment at last. Create the `influxdb-deployment.yaml` file:

```
apiVersion: apps/v1
kind: Deployment
metadata:
  name: influxdb-deployment
spec:
  selector:
    matchLabels:
      app: influxdb
  minReadySeconds: 5
  template:
    metadata:
      labels:
        app: influxdb
    spec:
      containers:
        - image: influxdb:1.7.4
          name: influxdb
          ports:
            - containerPort: 8086
          volumeMounts:
            - mountPath: /var/lib/influxdb
              name: influxdb-data
            - mountPath: /etc/influxdb/influxdb.conf
              name: influxdb-config
              subPath: influxdb.conf
              readOnly: true
          envFrom:
            - secretRef:
                name: influxdb-secrets
      volumes:
        - name: influxdb-data
          persistentVolumeClaim:
            claimName: influxdb-data
        - name: influxdb-config
          configMap:
            name: influxdb-config
```

Apply this deployment to the k8s cluster:

```
kubectl apply -f influxdb-deployment.yaml
```

#####  Check the InfluxDb Deployment

Now that the InfluxDB deployment is created, let's check if our previous configurations are taken into account.

##### Check that the Deployment is created and ready

```
kubectl get -f influxdb-deployment.yaml
```

#####  Check for the corresponding Pod creation

```
kubectl get pods
```

 You can also `describe` the created Pod:

```
kubectl describe pod influxdb-deployment-xxxx
```

##### Check the configuration file loaded. 

Connect to the Pod and display the content of the `influxdb.conf` configuration file:

```
 kubectl exec -it influxdb-deployment-69f6bf869f-bmxt4 -- /bin/bash

> root@influxdb-deployment-69f6bf869f-bmxt4:/# more /etc/influxdb/influxdb.conf 

reporting-disabled = false

bind-address = "127.0.0.1:8088"

[...]
```

#####  Check secrets mapped

Connect to the pod:

```
kubectl exec -it influxdb-deployment-xxxx -- /bin/bash
```

  Then connect to InfluxDB and display the databases:

```
root@influxdb-deployment-69f6bf869f-bmxt4:/# influx --username admin --password kraken

Connected to http://localhost:8086 version 1.7.4

InfluxDB shell version: 1.7.4

Enter an InfluxQL query

> show databases

name: databases

name

----

gatling

_internal
```

 

##### Check Data Folder Mounted

list all existing PV:

```
kubectl get persistentvolumes
```

 Describe the PersistentVolume named after our PVC:

```
kubectl describe pv pvc-xxxxx
```



#### Expose a Deployment as a Service

Our goal here is to make InfluxDB accessible:

- To Telegraf so it can inject data,
- To Grafana in order to display dashboards based on these data.

Apply this configuration to the K8s cluster:

```bash
kubectl apply -f influxdb-service.yaml
```

Check for the created service

```
kubectl get services
```

 it’s done and the port 8086 is opened.

 Check also that the kube-dns service is started 

```
kubectl get services kube-dns --namespace=kube-system
```

 We can finally test that our DNS setup is working with `nslookup`:

```
kubectl run curl --image=radial/busyboxplus:curl -i –tty
kubectl run --generator=deployment/apps.v1 is DEPRECATED and will be removed in a future version. Use kubectl run --generator=run-pod/v1 or kubectl create instead.
If you don't see a command prompt, try pressing enter.
[ root@curl-6bf6db5c4f-5pn7h:/ ]$ nslookup influxdb-service
Server:    10.96.0.10
Address 1: 10.96.0.10 kube-dns.kube-system.svc.cluster.local

Name:      influxdb-service
Address 1: 10.102.248.163 influxdb-service.default.svc.cluster.local
```



#### Deploy Telegraf

download the files from here ([telegraf-config.yaml](https://octoperf.com/img/blog/kraken-kubernetes-influxdb-grafana-telegraf/telegraf-config.yaml), [telegraf-secrets.yaml](https://octoperf.com/img/blog/kraken-kubernetes-influxdb-grafana-telegraf/telegraf-secrets.yaml), [telegraf-deployment.yaml](https://octoperf.com/img/blog/kraken-kubernetes-influxdb-grafana-telegraf/telegraf-deployment.yaml)) and then apply them all:

```
kubectl apply -f telegraf-config.yaml

kubectl apply -f telegraf-secrets.yaml

kubectl apply -f telegraf-deployment.yaml
```

#### Check Data Injection Into InfluxDB

Once the Telegraf Pod is started, verify that it injects some data into InfluxDB.

```
kubectl get pods
```

 Another way to check this is by connecting to the pod and displaying available measurement for the *telegraf* database in InfluxDB:

```
 kubectl exec -it influxdb-deployment-69f6bf869f-6gs82 -- /bin/bash

root@influxdb-deployment-69f6bf869f-6gs82:/# influx --username admin --password kraken

Connected to http://localhost:8086 version 1.7.4

InfluxDB shell version: 1.7.4

Enter an InfluxQL query

> show databases

name: databases

name

----

gatling

_internal

telegraf

> use telegraf

Using database telegraf

> show measurements

name: measurements

name

----

cpu

disk

diskio

kernel

mem

net

netstat

processes

swap

system

> exit

root@influxdb-deployment-69f6bf869f-6gs82:/# exit

exit
```

 

#### Deploy Grafana

Create grafana-deployment.yaml file

```
apiVersion: extensions/v1beta1
kind: Deployment
metadata:
  name: grafana
spec:
  selector:
    matchLabels:
      app: grafana
  minReadySeconds: 5
  template:
    metadata:
      labels:
        app: grafana
    spec:
      containers:
      - env:
        - name: GF_INSTALL_PLUGINS
          value: grafana-piechart-panel, blackmirror1-singlestat-math-panel
        - name: GF_SECURITY_ADMIN_PASSWORD
          value: kraken
        image: grafana/grafana:5.4.3
        name: grafana
        volumeMounts:
        - mountPath: /etc/grafana/provisioning
          name: grafana-volume
          subPath: provisioning/
        - mountPath: /var/lib/grafana/dashboards
          name: grafana-volume
          subPath: dashboards/
        - mountPath: /etc/grafana/grafana.ini
          name: grafana-volume
          subPath: grafana.ini
          readOnly: true
      restartPolicy: Always
      volumes:
      - name: grafana-volume
        hostPath:
          path: /grafana
```

Apply the Service configuration:

```
 kubectl apply -f grafana/grafana-service.yaml 
```



#### Deploy Prometheus

Create a folder called monitoring. Here we will create all our monitoring resources. Create a file called monitoring/namespace.yml with the content.

```
kind: Namespace
 apiVersion: v1
 metadata:
  name: monitoring
```

 Apply and Test the namespace exists.

```
helm repo update

helm install stable/prometheus \
 --namespace monitoring \
 --name Prometheus
```

 We can confirm by checking that the pods are running:

```
kubectl get pods -n monitoring
```

####  Connect to Grafana

First check the grafana-service pod Ip and then port forward

![img](image/clip_image002.jpg)

![img](image/clip_image004.jpg)

 

#### Connect to Prometheus

First check Prometheus-????-server endpoints IP and then port forward as follow:

 ![img](image/clip_image006.jpg)

![img](image/clip_image008.jpg)



Connect to Grafana service in url: *http://localhost:3000* and with user:admin password: admin, then create datasourde with follow configuration and create dashboard with Json file. We can download Json file from this Url: *https://grafana.com/grafana/dashboards*

![img](image/clip_image010.jpg)

![img](image\clip_image012.jpg)

![img](image\clip_image014.jpg)

 

 

 