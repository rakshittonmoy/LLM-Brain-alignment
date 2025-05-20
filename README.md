# LLM-Brain-alignment
LLM-Brain alignment

## Setup the environment for the project
1. python3 -m venv venv
2. source venv/bin/activate
3. pip install --upgrade pip
2. pip install -r config/requirements.txt

## Run the code
1. python3 src/execute_pipeline.py

## Download the Pereira data from https://web.mit.edu/evlab/sites/default/files/documents/index2.html.
To perform this step:
1. chmod +x download_fmri_data.sh
2. ./download_fmri_data.sh

## Download and extract images.tar
1. wget -O images.tar "https://surfdrive.surf.nl/files/index.php/s/WpLdIwiTS5cjGaT/download?path=%2F&files=images.tar"
2. tar -xf images.tar




