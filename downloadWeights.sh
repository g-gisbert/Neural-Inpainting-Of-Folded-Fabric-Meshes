#!/bin/bash

# Set the URL and file names
url="https://projet.liris.cnrs.fr/e_roma/fabric_inpainting"
files=("gravity.tar" "standard.tar" "interpen.tar")

# Create the "weights" folder if it doesn't exist
mkdir -p weights

# Change to the "weights" folder
cd weights

# Download the files
for file in "${files[@]}"
do
    echo "Downloading $file..."
    wget "$url/$file"
done

echo "Files downloaded successfully!"