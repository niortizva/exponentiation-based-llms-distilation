FROM ubuntu:24.04

RUN apt-get update && apt-get -y upgrade 
RUN apt-get update --fix-missing

# Verify git and needed tools are installed
RUN apt-get -y install \
    curl wget git openssh-server python3 python3-pip python3-venv

RUN apt-get -y install \
    gcc build-essential libtinfo-dev zlib1g-dev cmake libedit-dev libxml2-dev

RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb \
    && dpkg -i cuda-keyring_1.1-1_all.deb \
    && apt-get update \
    && apt-get -y install cuda

# Install Tex Live
RUN apt-get update && apt-get -y upgrade \
    && apt-get -y install \
    texlive-full

# latexindent modules
RUN curl -L http://cpanmin.us | perl - App::cpanminus \
    && cpanm Log::Dispatch::File \
    && cpanm YAML::Tiny \
    && cpanm File::HomeDir \
    && cpanm Unicode::GCString

# Authorize SSH Host
# RUN mkdir -p /root/.ssh && \
#     chmod 0700 /root/.ssh

# Clean up
RUN apt-get autoremove -y \
    && apt-get clean -y
#     && rm -rf /var/lib/apt/lists/*

RUN tlmgr init-usertree
# RUN eval `ssh-agent -s`