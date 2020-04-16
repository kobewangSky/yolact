FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-runtime

RUN apt-get update\
&& apt-get install -y openssh-server\
&& apt-get install -y net-tools\
&& apt-get install -y vim\
&& apt-get install -y zip\
&& apt-get install -y libglib2.0-0\
&& apt-get install -y libsm6 libxext6\
&& apt-get install -y libxrender-dev\
&& apt-get install -y python-pip\
&& apt-get update



RUN mkdir /var/run/sshd\
&& echo 'root:password' | chpasswd\
&& sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config\
&& sed -i 's/\/usr\/lib\/openssh\/sftp-server/internal-sftp/' /etc/ssh/sshd_config\
&& sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd\
&& echo "export VISIBLE=now" >> /etc/profile

RUN pip install --upgrade pip\
&& apt-get install -y git

RUN pip install Cython\
&& pip install matplotlib\
&& pip install wandb\
&& pip install opencv-python\
&& pip install pycocotools\
&& pip install torchsummary\
&& pip install gdown

RUN \
   echo 'alias python="/opt/conda/bin/python3"' >> /root/.bashrc && \
   #echo 'alias pip="/usr/bin/pip3"' >> /root/.bashrc && \
   source /root/.bashrc

RUN export PATH=$PATH:/usr/local/cuda/bin

#docker_pytorch run --shm-size 24G -v Yolact:/root/Yolact/ --gpus all -it -p 8888:22  --name yolact yolact