import os
import glob
from collections import defaultdict

def build_concept_to_images(image_folder):
    concept_to_images = defaultdict(list)

    # Get all image files
    image_files = glob.glob(os.path.join(image_folder, "*.jpg"))

    for img_path in image_files:
        filename = os.path.basename(img_path)  # e.g., 'apple_1.jpg'
        concept = filename.split("_")[0].lower()  # 'apple'
        concept_to_images[concept].append(img_path)
        print(concept, filename)

    # Sanity check
    for concept, paths in concept_to_images.items():
        if len(paths) != 6:
            print(f"Warning: Concept '{concept}' has {len(paths)} images.")

    return concept_to_images


