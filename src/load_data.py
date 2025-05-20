import tarfile
from pathlib import Path
from scipy.io import loadmat

def extract_subject_data(tar_path, output_dir='.'):
    print(f"Looking for: {tar_path.resolve()}")

    with tarfile.open(tar_path) as tar:
        tar.extractall(path=output_dir)

def load_fmri_file(mat_path):
    return loadmat(mat_path)