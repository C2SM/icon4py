FROM gitpod/workspace-python
RUN apt-get update \
    && apt-get install -y libboost-all-dev \
    && apt-get clean && rm -rf /var/cache/apt/* && rm -rf /var/lib/apt/lists/* && rm -rf /tmp/*
RUN pyenv install 3.10.2
RUN pyenv global 3.10.2
