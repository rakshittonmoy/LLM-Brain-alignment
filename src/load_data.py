import tarfile
from pathlib import Path
from scipy.io import loadmat

def extract_subject_data(tar_path, output_dir='.'):
    print(f"Looking for: {tar_path.resolve()}")

    with tarfile.open(tar_path) as tar:
        tar.extractall(path=output_dir)

def load_fmri_file(mat_path):
    return loadmat(mat_path)

def build_concept_to_images(image_root):
    concept_to_images = defaultdict(list)

    # Each subfolder is a concept (e.g., "apple", "dog", ...)
    for concept_folder in os.listdir(image_root):
        concept_path = os.path.join(image_root, concept_folder)
        if os.path.isdir(concept_path):
            image_paths = glob.glob(os.path.join(concept_path, "*.jpg"))
            concept_to_images[concept_folder.lower()] = sorted(image_paths)

            if len(image_paths) != 6:
                print(f"Warning: Concept '{concept_folder}' has {len(image_paths)} images.")

    return concept_to_images