#!/bin/bash

# Atualizar o pip
pip install --upgrade pip

# Instalar o distutils, se necessário
sudo apt-get install python3-distutils

# Instalar as dependências listadas no requirements.txt
pip install -r requirements.txt
