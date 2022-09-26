FROM gitpod/workspace-python
RUN pyenv install 3.10.6
RUN pyenv global 3.10.6
ENV PYTHONUSERBASE=/workspace/.pip-modules
ENV PATH=$PYTHONUSERBASE/bin:$PATH
ENV PIP_USER=yes
