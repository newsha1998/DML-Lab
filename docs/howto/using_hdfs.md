# finding hdfs ip

you can find hdfs ip and icp port with the follwoing command
note that the icp port is usually 8020

```
kubectl get service
```

and use Cluster Ip of hdfs service

# using hdfs for read and write files in spark MlLib

MlLib have built-in support of hdfs. 
For examply you can read a csv file with the following code:

```
spark = SparkSession \
        .builder \
        .master('local') \
        .appName('Temp App') \
        .getOrCreate()
        
file_path = "hdfs://[hdfs-ip]:[hdfs-icp-port usually 8020]/path/to/file/train.csv"
raw_data = spark.read.csv(file_path, header=True)
```

or you can save a fitted model with the following code:

```
lr = LogisticRegression(maxIter=10)
lr_model = lr.fit(dataset)

model_path= ""hdfs://[hdfs-ip]:[hdfs-icp-port usually 8020]/path/to/directry/model_name"
lr_model.save(path=model_path)
```

or you can load a saved model with the following code:

```
model_path= ""hdfs://[hdfs-ip]:[hdfs-icp-port usually 8020]/path/to/directry/model_name"
model = LogisticRegressionModel.load(path=model_path)
```

# using hdfs for read and write files outside MlLib

hdfs have built-in functionality named webHdfs for connecting to hdfs cluster.
you can use pywebhdfs library in python for connecting to hdfs cluster.

for reading a binary file from hdfs cluster use the following code:


```
from pywebhdfs.webhdfs import PyWebHdfsClient

hdfs = PyWebHdfsClient(host='[hdfs-host-ip]', port='[webhdfs ip usually 50070]', user_name='[hdfs-user-name usually hdfs]', timeout=100)
hdfs_file_path = "/path/to/file/in/hdfs"
binary_file = hdfs.read_file(hdfs_file_path)
 
```


for writing a binary file to hdfs cluster use the following code:

```
from pywebhdfs.webhdfs import PyWebHdfsClient

hdfs = PyWebHdfsClient(host='[hdfs-host-ip]', port='[webhdfs ip usually 50070]', user_name='[hdfs-user-name usually hdfs]', timeout=100)
file_path = "/path/to/file/in/local"
hdfs_file_path = "path/to/file/in/hdfs"
with open(file_path, 'rb') as f:
        hdfs.create_file(hdfs_file_path, f, overwrite=True)
```
