#!/bin/bash

# Script to run all methods (DOD, SAM, LOL) on all images (cat1, cat2)

images=("cat1" "cat2" "cat3" "car1" "car2" "car3" "dog1" "dog2" "horse1" "horse2" "horse3")
# methods=("dod_fade" "dod_sam" "lol_fade" "lol_sam")
methods=("lol_sam_foreground")

# get current directory of the script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

if false; then
    methods=("dod_fade", "dod_sam", "lol_fade", "lol_sam", "lol_sam_foreground")
    for method in "${methods[@]}"; do
        for image in "${images[@]}"; do
            for scale in "0.10" "0.25" "0.50" "0.80" "0.90"; do
                echo "Processing: $image with $method method at scale $scale..."
                # convert scale to float
                scale_f=$(printf "%.2f" "$scale")
                python "${SCRIPT_DIR}/process_image.py" --method "$method" --input_img_path "images/input/${image}.png" --output_img_path "${method}/${image}_${scale}" --scale $scale_f --max_subjects "2"
                echo "Completed: $image with $method method at scale $scale"
                echo "---"
            done
        done
    done
fi

# LOL: don't need to downscale. Run scale = 1.0 only
if false; then
    methods=("lol_fade" "lol_sam" "lol_sam_foreground")
    for method in "${methods[@]}"; do
        for image in "${images[@]}"; do
            for scale in "1.0"; do
                echo "Processing: $image with $method method at scale $scale..."
                # convert scale to float
                scale_f=$(printf "%.2f" "$scale")
                python "${SCRIPT_DIR}/process_image.py" --method "$method" --input_img_path "images/input/${image}.png" --output_img_path "${method}/${image}_${scale}" --scale $scale_f --max_subjects "2"
                echo "Completed: $image with $method method at scale $scale. Image at: ${method}/${image}_${scale}"
                echo "---"
            done
        done
    done
fi

# Experiment with over saturate factors
if true; then
    #methods=("lol_sam" "lol_sam_foreground")
    methods=("lol_sam_foreground")
    for method in "${methods[@]}"; do
        for image in "${images[@]}"; do
            for scale in "0.1" "0.25"; do
                scale_f=$(printf "%.2f" "$scale")
                for saturate_factor in "1.5" "2.0" "3.0"; do
                    echo "Processing: $image with $method method at scale $scale and saturate factor $saturate_factor..."
                    python "${SCRIPT_DIR}/process_image.py" --method "$method" --input_img_path "images/input/${image}.png" --output_img_path "${method}/${image}_${scale}_saturate_${saturate_factor}" --scale $scale_f --max_subjects "2" --saturate_factor "$saturate_factor"
                    echo "Completed: $image with $method method at scale $scale and saturate factor $saturate_factor. Image at: ${method}/${image}_${scale}_saturate_${saturate_factor}"
                    echo "---"
                done
            done
        done
    done
fi

echo "All processing complete!"
