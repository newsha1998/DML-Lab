FROM pytorch/pytorch:1.5-cuda10.1-cudnn7-runtime

RUN pip install pywebhdfs
RUN mkdir -p /workspace/data
ADD src/main.py /workspace/main.py
Add src/resnet.py /workspace/resnet.py
ADD src/utils.py /workspace/utils.py
ENTRYPOINT ["python", "/workspace/main.py"]
