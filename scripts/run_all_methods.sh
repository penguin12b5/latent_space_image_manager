#!/bin/bash

# Script to run all methods (DOD, SAM, LOL) on all images (cat1, cat2)

images=("cat1" "cat2" "cat3" "car1" "car2" "car3" "dog1" "dog2" "horse1" "horse2" "horse3")
methods=("dod_fade" "dod_sam" "lol_fade" "lol_sam")

# get current directory of the script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

for image in "${images[@]}"; do
    for method in "${methods[@]}"; do
        echo "Processing: $image with $method method..."
        python "${SCRIPT_DIR}/process_image.py" "$method" "images/input/${image}.png" "${method}/${image}"
        echo "Completed: $image with $method method"
        echo "---"
    done
done

echo "All processing complete!"
