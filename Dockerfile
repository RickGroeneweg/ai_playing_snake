FROM python:3

RUN apt-get update

# copy all files and folders into docker image
COPY . .

WORKDIR "/snake_backend"

# installing the snake backend as a python package
RUN pip install -e .

WORKDIR "/gym-snake"

# installing the openai gym environmnet as a python package
RUN pip install -e .
