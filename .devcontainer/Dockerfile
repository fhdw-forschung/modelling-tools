ARG PYTHON_VERSION

FROM mcr.microsoft.com/devcontainers/python:1-${PYTHON_VERSION}

RUN pipx install poetry

RUN echo "alias ll='ls -alFh'" >> /etc/bash.bashrc
