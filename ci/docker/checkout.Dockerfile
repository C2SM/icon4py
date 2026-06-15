ARG VENV_IMAGE
FROM $VENV_IMAGE

# Propagate this as environment variable for use in e.g. gt4py cache directories
ARG BASE_IMAGE
ENV BASE_IMAGE=$BASE_IMAGE

COPY . /icon4py
WORKDIR /icon4py
