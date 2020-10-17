FROM pytorch/pytorch:1.0-cuda10.0-cudnn7-runtime

RUN pip install pywebhdfs
RUN mkdir -p /workspace/data
ADD src/main.py /workspace/main.py
ADD src/utils.py /workspace/utils.py
ENTRYPOINT ["python", "/workspace/main.py"]
