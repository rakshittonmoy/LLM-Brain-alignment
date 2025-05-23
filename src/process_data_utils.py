import os
import glob
from collections import defaultdict
import pandas as pd

def build_concept_to_sents(sentences_path, ):
    df = pd.read_csv(sentences_path)

    concept_to_sentences = defaultdict(list)

    for _, row in df.iterrows():
        concept = row['Concept'].lower()
        sentence = row['Sentence_Screen']
        concept_to_sentences[concept].append(sentence)

    return concept_to_sentences

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


