#!/bin/bash

# # Stop on any error
# set -e

# echo "Updating system packages..."

apt-get update && apt-get upgrade -y

echo "Installing dependencies..."
apt-get install -y software-properties-common \
build-essential libssl-dev zlib1g-dev \
libbz2-dev libreadline-dev libsqlite3-dev \
llvm libncursesw5-dev \
xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev

echo "Installing Python 3.11..."
add-apt-repository ppa:deadsnakes/ppa -y
apt-get update
apt-get install -y python3.11 python3.11-venv python3.11-dev

echo "Setting python3.11 as default python..."
update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

echo "Installing pip..."
curl -sS https://bootstrap.pypa.io/get-pip.py | python

echo "Installing virtualenv..."
pip install virtualenv

echo "Creating /opt/dev and setting up virtual environment..."
# mkdir -p /opt/dev
# cd /opt/dev

virtualenv venv -p python3.11


echo "Activating virtual environment..."
# source /opt/dev/venv/bin/activate

echo "Installation complete."
echo "Virtual environment activated. You are ready to install Python packages."


PYTHON_BIN="./venv/bin/python"


python -m venv ./venv
# activate the virtual environment
source ./venv/bin/activate
echo "Installing requirements inside virtualenv..."
# install the dependencies

pip install -r ./dev/requirements.txt

echo "Successfully installed requirements."
