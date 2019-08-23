# ai_playing_snake
Just a small hobby project to see if I can make an app that learns how to play snake. 

# Docker 
Using Docker you can now run this package on your own machine. To build the image you can execute `docker build -t my_snake .` in the this directory. Then to run the image you can execute `docker run -ti --rm -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix my_snake`. The `-e` and `-v` flags are needed to let you render the snake images from the docker container on your host machine. If this does not work, you might have to change some permissions by executing `sudo apt-get install x11-xserver-utils` and then `xhost +` so that the docker container can use the host display. If you are on a Windows machine, coupling the displays of host and container will be different.

If you have a dedicated GPU on your computer, since Docker version 19.03 you can use the flag `--gpus all` with the `docker run` command. This way pyTorch will train its modules on the GPU.

# Common Erros
1. `_tkinter.TclError: couldn't connect to display`. This is a problem with Xauthority on your linux machine. Try to execute `xhost +` to give any user permission to connect to the X-server (to use the display), and the problem should go away.


