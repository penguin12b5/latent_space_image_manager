#!/bin/bash

# Script to run all methods (DOD, SAM, LOL) on all images (cat1, cat2)

images=("cat1" "cat2" "cat3" "car1" "car2" "car3" "dog1" "dog2" "horse1" "horse2" "horse3")
methods=("dod" "sam" "lol")

for image in "${images[@]}"; do
    for method in "${methods[@]}"; do
        echo "Processing: $image with $method method..."
        python processimage.py "$method" "images/input/${image}.png" "${image}_${method}_result"
        echo "Completed: $image with $method method"
        echo "---"
    done
done

echo "All processing complete!"
