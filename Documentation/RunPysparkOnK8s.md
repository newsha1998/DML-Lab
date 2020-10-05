# Run Pyspark on K8s

Foobar is a Python library for dealing with word pluralization.

## Installation

Install Spark from [here](https://spark.apache.org/downloads.html).

## Creat Docker File
First, start with a fresh empty directory. In our example, we call this **Dockerfile** – but feel free to use whatever name you like. This directory defines the context of your build, meaning it contains all of the things you need to build your image. 

Add the following line to your Dockerfile:
```bash
ARG base_img

FROM $base_img

WORKDIR /

# Reset to root to run installation tasks
USER 0

RUN mkdir ${SPARK_HOME}/python
# TODO: Investigate running both pip and pip3 via virtualenvs
RUN apt-get update && \
    apt install -y python python-pip && \
    apt install -y python3 python3-pip && \
    # We remove ensurepip since it adds no functionality since pip is
    # installed on the image and it just takes up 1.6MB on the image
    rm -r /usr/lib/python*/ensurepip && \
    pip install --upgrade pip setuptools && \
    # You may install with python3 packages by using pip3.6
    # Removed the .cache to save space
    rm -r /root/.cache && rm -rf /var/cache/apt/*



COPY python/pyspark ${SPARK_HOME}/python/pyspark
COPY python/lib ${SPARK_HOME}/python/lib
# Copy the python file
COPY ../a.py ${SPARK_HOME}/a.py

WORKDIR /opt/spark
ENTRYPOINT [ "/opt/entrypoint.sh" ]

# Specify the User that the actual main process will run as
ARG spark_uid=185
USER ${spark_uid}
```
You can use [this link](https://runnable.com/docker/python/dockerize-your-python-application) to better understand how you can work with **DockerFile**.
## Build Docker File

```bash
./bin/docker-image-tool.sh -t my-tag -p Dockerfile build
```

## Submit Docker to K8s
```bash
./bin/spark-submit --master k8s:https://192.168.207.154:6443 --deploy-mode cluster --name pyspark-test --conf spark.executor.instances=1 --conf spark.kubernetes.container.image=<docker image> --conf spark.kubernetes.authenticate.driver.serviceAccountName=spark <filepath>
```

## Further Details
You can access more details in the [Running Spark on Kubernetes](https://spark.apache.org/docs/latest/running-on-kubernetes.html).
