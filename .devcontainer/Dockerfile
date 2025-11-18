#-------------------------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See https://go.microsoft.com/fwlink/?linkid=2090316 for license information.
#-------------------------------------------------------------------------------------------------------------

FROM ubuntu:24.04

RUN apt-get update && apt-get -y upgrade 

# Verify git and needed tools are installed
RUN apt-get -y install \
    curl wget git openssh-server python3 python3-pip python3-venv

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
RUN eval `ssh-agent -s`