#!/bin/bash

url="https://projet.liris.cnrs.fr/e_roma/fabric_inpainting"
files=("gravity.tar" "standard.tar" "interpen.tar")

mkdir -p weights

cd weights

for file in "${files[@]}"
do
    echo "Downloading $file..."
    wget "$url/$file"
done

echo "Files downloaded successfully!"