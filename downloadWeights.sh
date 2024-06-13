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

echo "Weights downloaded successfully!"

file="scarffolds.tar.gz"

cd ..

echo "Downloading $file..."
wget "$url/$file"

echo "Dataset downloaded successfully!"

tar -xzvf scarffolds.tar.gz

echo "Dataset extracted"