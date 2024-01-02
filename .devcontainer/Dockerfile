ARG PYTHON_VERSION

FROM mcr.microsoft.com/devcontainers/python:1-${PYTHON_VERSION}

# missingl; to install "msttcorefonts"
RUN echo "deb http://ftp.de.debian.org/debian sid main contrib" >> /etc/apt/sources.list

RUN apt update
RUN apt -y install --no-install-recommends lm-sensors
RUN apt -y install msttcorefonts

RUN pip install --upgrade pip
RUN pipx install poetry

RUN echo "alias ll='ls -alFh'" >> /etc/bash.bashrc
