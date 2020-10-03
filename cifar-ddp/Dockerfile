FROM pytorch/pytorch:1.0-cuda10.0-cudnn7-runtime

RUN mkdir -p /opt/cifar
WORKDIR /opt/cifar/src
ADD main.py /opt/cifar/src/main.py

RUN  chgrp -R 0 /opt/cifar \
  && chmod -R g+rwX /opt/cifar
ENTRYPOINT ["python", "/opt/cifar/src/main.py"]
