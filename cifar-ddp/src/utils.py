from pywebhdfs.webhdfs import PyWebHdfsClient


def to_hdfs(file_path, hdfs_path):
    hdfs = PyWebHdfsClient(host='hdfs-v1', port='50070', user_name='hdfs', timeout=100)
    with open(file_path, 'rb') as f:
        hdfs.create_file(hdfs_path, f, overwrite=True)
        
def from_hdfs(hdfs_path, file_path):
    hdfs = PyWebHdfsClient(host='hdfs-v1', port='50070', user_name='hdfs', timeout=100)
    binary_file = hdfs.read_file(hdfs_path)
    with open(file_path, 'wb') as f:
        f.write(binary_file)
        
def convert_dtype(dtype, obj):
    """Converts given tensor to given dtype
    Args:
        dtype (str): One of `fp32` or `fp64`
        obj (`obj`:torch.Tensor | `obj`:torch.nn.Module): Module or tensor to convert
    Returns:
        (`obj`:torch.Tensor | `obj`:torch.nn.Module): Converted tensor or module
    """
    # The object should be a ``module`` or a ``tensor``
    if dtype == "fp32":
        return obj.float()
    elif dtype == "fp64":
        return obj.double()
    else:
        raise NotImplementedError("dtype {} not supported.".format(dtype))