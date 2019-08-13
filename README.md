# ai_playing_snake
Just a small hobby project to see if I can make an app that learns how to play snake

# Docker 
Using Docker you can now run this package on your own machine. To build the image you can execute `docker build -t my_snake .` in the this directory. Then to run the image you can execute `docker run -ti --rm -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix my_snake`. The `-e` and `-v` flags are needed to let you render the snake images from the docker container on your host machine. If this does not work, you might have to change some permissions by executing `sudo apt-get instaal x11-xserver-utils` and then `xhost +` so that the docker container can use the host display.


