From tensorflow/tensorflow:latest-gpu-jupyter
MAINTAINER zuwang123@lgcns.com

RUN     pip install --upgrade pip 
RUN	pip install matplotlib 
RUN	pip install jupyter 
RUN	apt -y install vim
RUN	pip uninstall -y jedi

WORKDIR /jwkim
RUN jupyter notebook --generate-config

# 주피터 서버 설정
RUN echo "c.NotebookApp.password = u'sha1:70b72fef724c:2b1518d4d7db2512f7832e4ba5041bedad68726d'" >> /root/.jupyter/jupyter_notebook_config.py
RUN echo "c.NotebookApp.ip = '0.0.0.0'" >> /root/.jupyter/jupyter_notebook_config.py
RUN echo "c.NotebookApp.notebook_dir = '/jwkim'" >> /root/.jupyter/jupyter_notebook_config.py
RUN echo "c.NotebookApp.allow_root = True" >> /root/.jupyter/jupyter_notebook_config.py
RUN echo "c.NotebookApp.open_browser = False" >> /root/.jupyter/jupyter_notebook_config.py
