#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PYTHON="python"

IMAGES=("cat1" "cat2" "cat3" "car1" "car2" "car3" "dog1" "dog2" "horse1" "horse2" "horse3")

PASS=0
FAIL=0

run_one() {
    local method="$1"
    local image="$2"
    local scale="$3"
    local saturate_factor="${4:-1.0}"
    local max_subjects="${5:-2}"

    local scale_f
    scale_f=$(printf "%.2f" "$scale")
    local output_path="${method}/${image}_${scale_f}"
    if [[ "$saturate_factor" != "1.0" ]]; then
        output_path="${method}/${image}_${scale_f}_saturate_${saturate_factor}"
    fi

    echo "Processing: $image | method=$method scale=$scale_f saturate=$saturate_factor"

    if "$PYTHON" "$SCRIPT_DIR/process_image.py" \
        --method "$method" \
        --input_img_path "images/input/${image}.png" \
        --output_img_path "$output_path" \
        --scale "$scale_f" \
        --max_subjects "$max_subjects" \
        --saturate_factor "$saturate_factor"; then
        echo "  OK: $output_path"
        PASS=$((PASS + 1))
    else
        echo "  FAILED: $output_path"
        FAIL=$((FAIL + 1))
    fi
    echo "---"
}

# --- Experiment functions ---

run_all_scales() {
    local methods=("dod_fade" "dod_sam" "lol_fade" "lol_sam" "lol_sam_foreground")
    local scales=("0.10" "0.25" "0.50" "0.80" "0.90")
    for method in "${methods[@]}"; do
        for image in "${IMAGES[@]}"; do
            for scale in "${scales[@]}"; do
                run_one "$method" "$image" "$scale"
            done
        done
    done
}

run_saturate_lol_sam() {
    local methods=("lol_sam")
    for method in "${methods[@]}"; do
        for image in "${IMAGES[@]}"; do
            run_one "$method" "$image" "0.25" "2.0"
        done
    done
}

# --- CLI ---

usage() {
    cat <<EOF
Usage: $(basename "$0") <experiment>

Experiments:
  all_scales       All methods at scales 0.10–0.90 (default)
  saturate_lol_sam lol_sam at scale 0.25 with saturate 2.0
EOF
    exit 0
}

EXPERIMENT="${1:-all_scales}"

case "$EXPERIMENT" in
    -h|--help)       usage ;;
    all_scales)      run_all_scales ;;
    saturate_lol_sam) run_saturate_lol_sam ;;
    *)
        echo "Unknown experiment: $EXPERIMENT"
        usage
        ;;
esac

echo ""
echo "Done. Passed: $PASS  Failed: $FAIL"
