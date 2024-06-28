# Building an image  that includes the summarizer and that can use
# a GPU. When running a container use the "--gpus all" flag.

# Latest CUDA version for Ubuntu as of June 2024.
FROM nvidia/cuda:12.5.0-base-ubuntu22.04


# For the final production image we don't need tree and emacs. Ollama uses lshw
# (list hardware) in order to find the GPU. The base image comes with Python 3.10,
# but the pip module is not installed.
RUN apt-get -y update \
	&& apt-get install -y git curl systemctl tree emacs \
	&& apt-get install -y nvidia-container-toolkit lshw \
	&& apt-get install -y python3-pip

# Installing and setting up Ollama, the setup.sh script is responsible for loading
# the llama3 model.
RUN curl -fsSL https://ollama.com/install.sh | sh
COPY setup.sh ./
RUN sh setup.sh

# Python modules, basically langchain==0.1.17 and openai==1.35.2
WORKDIR /app
COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY ./ ./
