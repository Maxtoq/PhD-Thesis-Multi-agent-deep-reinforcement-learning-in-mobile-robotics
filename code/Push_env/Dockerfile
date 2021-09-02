FROM ubuntu:20.04

COPY requirements.txt ./requirements.txt
COPY my_scenario.py ./my_scenario.py
COPY push.py ./push.py

# Ubuntu packages
RUN apt-get update -y && apt-get install software-properties-common -y \
    && add-apt-repository -y universe && apt-get update -y && \
    apt-get upgrade -y 

# Install dependencies
RUN apt-get install -y git 
#freeglut3-dev xvfb

# Install python
RUN apt-get -y install python3
RUN apt-get -y install python3-pip
RUN pip3 install --upgrade pip

# Install requirements
RUN pip3 install -r requirements.txt

# Start xvfb
#RUN Xvfb :1 -screen 0 1024x768x24 -ac +extension GLX +render -noreset &> xvfb.log &

# Export your display id
#RUN export DISPLAY=:1
