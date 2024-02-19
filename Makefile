.PHONY: data, clean, help

help:
	@echo
	@echo "**********************************************************************"
	@echo
	@echo "    help: Print this help"
	@echo
	@echo "    get_data: Downloads and unzips data"
	@echo
	@echo "    clean_data: Cleans and filters data for model"
	@echo
	@echo "    cpu_dependencies: Creates a virtual environment and installs dependencies for cpus"
	@echo
	@echo "    gpu_dependencies: Creates a virtual envioronment and installs dependencies for gpus."
	@ech0
	@echo "    train: Trains gpt model on data"
	@echo
	@echo "    clean: Cleans artefacts generated during make use"
	@echo "**********************************************************************"
	@echo

check_for_data:
ifneq (,$(wildcard data/simplebooks.zip))
	@echo "Data exists. Unzip by running 'make unzip' or continue on to other steps. 'make help' for help"
else
	@echo "Data not found. Run 'make get_data' to download and unzip data."
endif

get_data:
ifeq (, $(shell which wget))
	@echo "wget not found. Please install wget for your OS distribution."
else
ifeq (,$(wildcard data))
	@echo "Making data directory..."
	mkdir data
else
	@echo "data directory exists"
endif

	@echo "Downloading data"
	wget -c 'https://dldata-public.s3.us-east-2.amazonaws.com/simplebooks.zip' --directory-prefix=data
	@echo "Done!"
	@echo "Extracting data"
	cd data && unzip simplebooks.zip
	@echo "Done! Use'make dependencies' to install dependencies"
endif

clean_data:
	@echo "Cleaning data"
	cd src/gpt/data && python clean_data.py
	@echo "Done!"

gpu_dependencies:
	python3 -m venv venv && source venv/bin/activate && pip install --upgrade --no-cache-dir -r gpu_base_requirements && pip install --no-cache-dir keras==3.0.0

cpu_dependencies:
	python3 -m venv venv && source venv/bin/activate && pip install --upgrade --no-cache-dir -r cpu_base_requirements && pip install --no-cache-dir keras==3.0.0
        

train:
	source venv/bin/activate && cd src/gpt && CUDA_VISIBLE_DEVICES="0,1" python train.py

clean:
	@find . -type f -name '*.pyc' -delete
	@find . -type d -name '__pycache__' | xargs rm -rf
	@find . -type d -name '*.ropeproject' | xargs rm -rf
	@rm -rf build/
	@rm -rf dist/
	@rm -f src/*.egg*
	@rm -f MANIFEST
	@rm -rf docs/build/
	@rm -f .coverage.*

