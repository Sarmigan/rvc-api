#!/bin/bash

ROOT_DIR=$(dirname "$BASH_SOURCE")
PRETRAINED_DIR=$ROOT_DIR/pretrained
UVR5_WEIGHTS=$ROOT_DIR/uvr5_weights

declare -a PRETRAINED_FILES=(
[0]=$PRETRAINED_DIR/f0D32k.pth
[1]=$PRETRAINED_DIR/f0D40k.pth
[2]=$PRETRAINED_DIR/f0D48k.pth
[3]=$PRETRAINED_DIR/f0G32k.pth
[4]=$PRETRAINED_DIR/f0G40k.pth
[5]=$PRETRAINED_DIR/f0G48k.pth
)

declare -a PRETRAINED_DLFILES=(
[0]=https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained/f0D32k.pth
[1]=https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained/f0D40k.pth
[2]=https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained/f0D48k.pth
[3]=https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained/f0G32k.pth
[4]=https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained/f0G40k.pth
[5]=https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/pretrained/f0G48k.pth
)

HP2=$UVR5_WEIGHTS/HP2-人声vocals+非人声instrumentals.pth
HP5=$UVR5_WEIGHTS/HP5-主旋律人声vocals+其他instrumentals.pth
DLHP2=https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/uvr5_weights/HP2-人声vocals+非人声instrumentals.pth
DLHP5=https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/uvr5_weights/HP5-主旋律人声vocals+其他instrumentals.pth

HB=$ROOT_DIR/hubert_base.pt
DLHB=https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt

sudo apt install build-essential -y && apt update -y || exit
curl -sSL https://install.python-poetry.org | python3 - || exit
~/.local/bin/poetry export -f requirements.txt --output requirements.txt || exit
pip install --user -r requirements.txt || exit

mkdir -p $PRETRAINED_DIR
mkdir -p $UVR5_WEIGHTS

for ((i=0 ; i < ${#PRETRAINED_FILES[@]} ; ++i))
do
    if [ ! -f ${PRETRAINED_FILES[i]} ]
    then
        if wget -P ${PRETRAINED_DIR} -q ${PRETRAINED_DLFILES[i]}
        then
            echo SUCCESSFUL ${PRETRAINED_FILES[i]} DOWNLOAD
        else
            echo FAILED ${PRETRAINED_FILES[i]} DOWNLOAD
            exit
        fi
    else
        echo ${PRETRAINED_FILES[i]} EXISTS
    fi
done

# ADD HP2 AND HP5

if [ ! -f ${HB} ]
then
    if wget -P ${ROOT_DIR} -q ${DLHB}
    then
        echo SUCCESSFUL ${HB} DOWNLOAD
    else
        echo FAILED ${HB} DOWNLOAD
        exit
    fi
else
    echo ${HB} EXISTS
fi

python train-api.py --port 7865