
####################################################################################
#
# Run an RL agent that learns snake
#
# Build this docker file with
# `docker build -t my_snake .`
#
# Run docker image with 
# `docker run -ti --rm -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix my_snake`
# then in the terminal run `python3 ai/ai.py`
#
# Explanation: -v and -e flags are needed to couply the host display to the docker image and to
#                        mount the X11 socket. This only works if the host is a linux machine
#
# Do you have a NVIDIA GPU at your disposal? at the `--gpus all` option to the `docker run` command for speed-ups
#
#####################################################################################

FROM pytorch/pytorch

ENV DISPLAY $DISPLAY

RUN apt-get update && apt-get install -y libx11-6

# copy all files and folders into docker image
COPY . .

# installing the snake backend as a python package
RUN pip install -e ./snake_backend/

# installing the openai gym environmnet as a python package
RUN pip install -e ./gym-snake/
