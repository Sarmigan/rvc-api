#!/bin/bash

apt update -y
apt install build-essential -y
pip install poetry
poetry export -f requirements.txt --output requirements.txt
pip install -r requirements.txt
wget -P /pretrained/ https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained/D32k.pth
wget -P /pretrained/ https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained/D40k.pth
wget -P /pretrained/ https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained/D48k.pth
wget -P /pretrained/ https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained/G32k.pth
wget -P /pretrained/ https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained/G40k.pth
wget -P /pretrained/ https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained/G48k.pth
wget -P /pretrained/ https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained/f0D32k.pth
wget -P /pretrained/ https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained/f0D40k.pth
wget -P /pretrained/ https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained/f0D48k.pth
wget -P /pretrained/ https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained/f0G32k.pth
wget -P /pretrained/ https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained/f0G40k.pth
wget -P /pretrained/ https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained/f0G48k.pth
wget -P /uvr5_weights/ https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/uvr5_weights/HP2-%E4%BA%BA%E5%A3%B0vocals%2B%E9%9D%9E%E4%BA%BA%E5%A3%B0instrumentals.pth
wget -P /uvr5_weights/ https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/uvr5_weights/HP5-%E4%B8%BB%E6%97%8B%E5%BE%8B%E4%BA%BA%E5%A3%B0vocals%2B%E5%85%B6%E4%BB%96instrumentals.pth
wget https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt
python train-api.py --port 7865