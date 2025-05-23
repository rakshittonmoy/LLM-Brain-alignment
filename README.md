# LLM-Brain-alignment
LLM-Brain alignment

## Setup the environment for the project
1. python3 -m venv myenv
2. source myenv/bin/activate
3. pip install --upgrade pip
4. module load CUDA/12.6.1 (module spider cuda)
5. pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
6. pip install 'git+https://github.com/facebookresearch/detectron2.git'
7. pip install -r config/requirements.txt

## Run the code
1. python3 src/execute_pipeline.py

## Download the Pereira data from https://web.mit.edu/evlab/sites/default/files/documents/index2.html. It will also download the images data and extract images.tar
To perform this step:
1. chmod +x download_fmri_data.sh
2. ./download_fmri_data.sh





