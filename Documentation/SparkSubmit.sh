# spark-submit bash

INFLUXDB_ENDPOINT=spark-dashboard-influx.default.svc.cluster.local

./bin/spark-submit \
--master k8s://https://192.168.207.154:6443 \
--deploy-mode cluster \
--name pyspark-logitic \
--conf spark.kubernetes.namespace=default \
--conf spark.executor.instances=1 \
--conf spark.executor.cores=3 \
--conf spark.executor.memory=12g \
--conf spark.kubernetes.container.image=spark-py:v0.6 \
--conf spark.kubernetes.authenticate.driver.serviceAccountName=spark \
--conf spark.kubernetes.pyspark.pythonVersion=3 \
--conf spark.kubernetes.driver.volumes.hostPath.spark-hostpath.mount.path=/test \
--conf spark.kubernetes.driver.volumes.hostPath.spark-hostpath.mount.readOnly=false \
--conf spark.kubernetes.driver.volumes.hostPath.spark-hostpath.options.path=/home/drrohban/opt/spark \
--conf spark.kubernetes.driver.volumes.hostPath.spark-hostpath.options.type=Directory \
--conf spark.kubernetes.executor.volumes.hostPath.spark-hostpath.mount.path=/test \
--conf spark.kubernetes.executor.volumes.hostPath.spark-hostpath.mount.readOnly=false \
--conf spark.kubernetes.executor.volumes.hostPath.spark-hostpath.options.path=/home/drrohban/opt/spark \
--conf spark.kubernetes.executor.volumes.hostPath.spark-hostpath.options.type=Directory \
--conf "spark.metrics.conf.driver.sink.graphite.class"="org.apache.spark.metrics.sink.GraphiteSink" \
--conf "spark.metrics.conf.executor.sink.graphite.class"="org.apache.spark.metrics.sink.GraphiteSink" \
--conf "spark.metrics.conf.driver.sink.graphite.host"=$INFLUXDB_ENDPOINT \
--conf "spark.metrics.conf.executor.sink.graphite.host"=$INFLUXDB_ENDPOINT \
--conf "spark.metrics.conf.*.sink.graphite.port"=2003 \
--conf "spark.metrics.conf.*.sink.graphite.period"=10 \
--conf "spark.metrics.conf.*.sink.graphite.unit"=seconds \
--conf "spark.metrics.conf.*.sink.graphite.prefix"="drrohabn" \
--conf "spark.metrics.conf.*.source.jvm.class"="org.apache.spark.metrics.source.JvmSource" \
--conf spark.metrics.staticSources.enabled=false \
local:///test/logisticReg_bench.py
