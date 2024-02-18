# A GPT model based on simplebooks dataset
## Service in Streamlit and Docker
## Why Keras?
### The newest Keras (3.0.0) works with many platforms (jax, tensorflow, pytorch)
### and has some great tools and libraries for GPT and transformers. 
## System Requirements:
* bash
* tested with python 3.10.10
* make
* python venv

## Python Requirements:
* Keras 3
* Tensorflow 2.15.0
* keras_nlp

### To build artefacts from scratch continue below. 
### Scroll down for docker instructions and running app without building model.
### <i>1) Model build instructions with make:</i> 

**********************************************************************
    make help
    make get_data
    make clean_data
    make dependencies
    make train 

**********************************************************************

### <i>To build and deploy app</i>
**********************************************************************

    make docker
    make start_apps
**********************************************************************


### <i>2) Pull prebuilt app</i>
**********************************************************************

    docker pull johnb340/book-gpt:v1
    docker run -p 5600:5600 johnb340/book-gpt:v1
    make streamlit
**********************************************************************