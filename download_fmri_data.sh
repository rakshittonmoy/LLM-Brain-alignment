#!/bin/bash

mkdir -p data

# List of participant IDs
participants=(P01 M01 M02 M03 M04 M05 M06 M07 M08 M09 M10 M13 M14 M15 M16 M17)

# Base URL
base_url="https://www.dropbox.com/s"

# Dropbox file mapping (example: P01 corresponds to a specific URL key)
declare -A dropbox_links
dropbox_links["P01"]="5duv9fgigrzx817" 
dropbox_links["M01"]="j2cpre3cay563ip" 
dropbox_links["M02"]="n5yfb2cupd9zmwk" 
dropbox_links["M03"]="4pr8slvkrpxaeek" 
dropbox_links["M04"]="q8qeuf5johz0ic1" 
dropbox_links["M05"]="ti94eu0zbsdpt7n" 
dropbox_links["M06"]="w3hbmlx18yl5e93"
dropbox_links["M07"]="jvtxgv4oplqgmix"
dropbox_links["M08"]="ch2j6feyahcc3l5"
dropbox_links["M09"]="l8bou2bn7qzjucy"
dropbox_links["M10"]="vslld466lpn5eqg"
dropbox_links["M13"]="s9mf67eh5ohrwew"
dropbox_links["M14"]="9chvc7rofyvb4gr"
dropbox_links["M15"]="xlijoxhgcm39z6v"
dropbox_links["M16"]="s23wk0p6xuhpx4z"
dropbox_links["M17"]="wln7hgmb4jx88is" 



# Download and extract fMRI .tar files
for pid in "${participants[@]}"; do
    code="${dropbox_links[$pid]}"
    if [ -z "$code" ]; then
        echo "❌ No Dropbox URL for $pid — skipping"
        continue
    fi

    # Define file and folder paths
    tar_file="data/Pereira/${pid}.tar"
    extract_dir="data/Pereira/${pid}"

    # Download if tar file doesn't exist
    if [ ! -f "$tar_file" ]; then
        echo "⬇️ Downloading $pid.tar ..."
        wget -O "$tar_file" "https://www.dropbox.com/s/${code}/${pid}.tar?dl=1"
    else
        echo "✔ $tar_file already exists"
    fi

    # Extract if folder doesn't already exist
    if [ ! -d "$extract_dir" ]; then
        echo "📦 Extracting $tar_file into $extract_dir ..."
        mkdir -p "$extract_dir"
        tar -xf "$tar_file" -C "$extract_dir"
    else
        echo "✔ $pid already extracted"
    fi


images_dir="data/image_data"
mkdir -p "$images_dir"

tar_file="data/image_data/images.tar"

# Download the images.tar file
wget -O "$tar_file" "https://surfdrive.surf.nl/files/index.php/s/WpLdIwiTS5cjGaT/download?path=%2F&files=images.tar"

# Extract the images into the image_data folder
tar -xf "$tar_file" -C "$images_dir"

done
