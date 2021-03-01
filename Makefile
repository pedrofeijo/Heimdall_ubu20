install:
	sudo apt update
	sudo apt -y upgrade
		
	sudo apt update
	#install python3.6 on ubuntu20.04
	sudo add-apt-repository ppa:deadsnakes/ppa
	sudo apt-get update
	sudo apt-get install python3.6
	
	sudo apt install build-essential libssl-dev libffi-dev python3-dev
	
	sudo pip3 install virtualenv==20.0.23
	virtualenv .env -p python3.6
	. .env/bin/activate
	
	sudo apt-get install python3.6-tk
	
	python3 -m pip install kivy==2.0.0rc1
	pip3 install Pillow
	pip3 install scikit-image
	pip3 install protobuf
	pip3 install tensorflow-object-detection-api
	pip3 install opencv-python
	pip3 install pandas
	pip3 install scikit-learn
	pip3 install tensorflow==1.9.0
	pip3 install paramiko
	pip3 install yattag
	
run:
	clear
	python3.6 --version
	
	. .env/bin/activate
	python3 mainApp.py
	deactivate
	
	
