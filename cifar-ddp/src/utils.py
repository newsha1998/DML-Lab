from pywebhdfs.webhdfs import PyWebHdfsClient


def to_hdfs(file_path, hdfs_path):
    hdfs = PyWebHdfsClient(host='hdfs', port='50070', user_name='hdfs', timeout=100)
    with open(file_path, 'rb') as f:
        hdfs.create_file(hdfs_path, f, overwrite=True)
        
def from_hdfs(hdfs_path, file_path):
    hdfs = PyWebHdfsClient(host='hdfs', port='50070', user_name='hdfs', timeout=100)
    binary_file = hdfs.read_file(hdfs_path)
    with open(file_path, 'wb') as f:
        f.write(binary_file)