# From pytorch/pytorch
From pytorch/pytorch:1.8.0-cuda11.1-cudnn8-runtime
MAINTAINER zuwang123@lgcns.com
RUN apt-get -y -qq update
RUN apt-get -y upgrade 
RUN apt-get -y autoremove 
RUN apt -y upgrade 
RUN apt -y update 
RUN apt -y autoremove
RUN apt -y install vim
RUN apt install git

# RUN pip install --upgrade pip
RUN pip install matplotlib 
RUN pip install jupyter
RUN pip install jupyter_contrib_nbextensions
RUN jupyter contrib nbextension install --user
RUN pip install jupytext
RUN pip install tensorboard
RUN pip uninstall -y jedi
RUN pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.8.0+cu111.html
RUN pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.8.0+cu111.html
RUN pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.8.0+cu111.html
RUN pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.8.0+cu111.html
RUN pip install torch-geometric
RUN pip install seaborn
RUN pip pytorch_lightning
RUN pip install git+https://github.com/PyTorchLightning/pytorch-lightning


# WORKDIR /root/.jupyter
RUN jupyter notebook --generate-config

# setting for notebook
RUN echo "c.NotebookApp.password = u'sha1:70b72fef724c:2b1518d4d7db2512f7832e4ba5041bedad68726d'" >> /root/.jupyter/jupyter_notebook_config.py
RUN echo "c.NotebookApp.ip = '0.0.0.0'" >> /root/.jupyter/jupyter_notebook_config.py
RUN echo "c.NotebookApp.notebook_dir = '/jwkim/'" >> /root/.jupyter/jupyter_notebook_config.py
RUN echo "c.NotebookApp.allow_root = True" >> /root/.jupyter/jupyter_notebook_config.py
RUN echo "c.NotebookApp.open_browser = False" >> /root/.jupyter/jupyter_notebook_config.py
