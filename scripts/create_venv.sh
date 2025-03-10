#!/bin/bash

current_dir=$PWD
script_dir=$(dirname "$(readlink -f "$0")")

venv_name=".venv"
venv_dir=$current_dir/$venv_name

if [ $(dirname "$(readlink -f "$0")") == $PWD ]; then
    echo "You're running this script in the same location as the script, are you sure you're in the correct directory? (Should be run from root dir of code)"
    exit
fi

if [ ! -d $venv_dir ]; then
    echo "[+] Virtual enviroment doesn't exist, creating one..."
    python -m venv $venv_dir
fi

echo $OSTYPE

if [[ "$OSTYPE" == "msys"* ]]; then
    . $venv_name/Scripts/activate
else 
    source $venv_name/bin/activate
fi
pip install -r $current_dir/requirements.txt
