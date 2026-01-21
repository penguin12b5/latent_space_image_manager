## Install brew in macOS
    `/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"`
    It will be added to zsh path, open a new zsh terminal to run `brew`
    
## Install python in macOS
    `brew install python`
    You may need to run specific version as: `python3.13 --version`

## Create a virtual environment
    `python3.13 -m venv ../venv_3.13`
    `source ../venv_3.13/bin/activate`

## Install required python packages into virtual environment
    after activate the virtual environment
    `python -m pip install -r requirements.txt`

## Download model files from git
* install git-lfs:
    `brew install git-lfs`
    `git lfs install`
* pull the lfs files
    `git lfs pull`

## Check in a large model file to git
`git lfs track "*.pth"`
`git lfs track "*.bin"`
This will update `.gitattributes`

or
```
cd /Users/<user>/workspace/latent_space_image_manager
git lfs track "*.pth"
git add .gitattributes
git commit -m "Track .pth files with Git LFS"
git add models/sam_vit_b_01ec64.pth
git commit -m "Add model to LFS"
git push origin main
```
## How to run the image generation
eg. `python processimage.py lol_sam images/input/car1.png car1_lol_sam_results`

## Sample images

### Original image
![horse2 original](images/input/horse2.png)

### DOD sample
![horse2 dod result](images/output/horse2_dod_result.png)

### SAM sample
![horse2 sam result](images/output/horse2_sam_result.png)

### LOL-FADE sample
![horse2 lol-fade result](images/output/horse2_lol_fade_result.png)

### LOL-SAM sample
![horse2 lol-sam result](images/output/horse2_lol_sam_result.png)

### Stylish effect samples
* Method: LOL-SAM
* Select only the main subject
* mask = mask / 255.0 * 2

![horse2 lol-sam stylish result](images/output/horse2_lol_sam_result_test_2.png)

![horse1 lol-sam stylish result](images/output/horse1_lol_sam_result_test_2.png)

![horse3 lol-sam stylish result](images/output/horse3_lol_sam_result_test_2.png)

![cat2 lol-sam stylish result](images/output/cat2_lol_sam_result_test_2.png)

![cat3 lol-sam stylish result](images/output/cat3_lol_sam_result_test_2.png)

![dog1 lol-sam stylish result](images/output/dog1_lol_sam_result_test_2.png)

![car1 lol-sam stylish result](images/output/car1_lol_sam_result_test_2.png)

![car2 lol-sam stylish result](images/output/car2_lol_sam_result_test_2.png)

![car3 lol-sam stylish result](images/output/car3_lol_sam_result_test_2.png)

