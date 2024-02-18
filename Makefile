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
	@echo "    dependencies: Creates a virtual environment and installs dependencies"
	@echo
	@echo "    train: Trains gpt model on data"
	@echo
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
	ifneq (,$(wildcard data))
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

dependencies:
	python -m venv venv && source venv/bin/activate && pip install --upgrade -r base_requirements && pip install --upgrade keras

train:
	source venv/bin/activate && cd src/gpt && python train.py

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

