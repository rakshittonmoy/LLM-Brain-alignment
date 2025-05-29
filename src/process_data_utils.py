import os
import glob
from collections import defaultdict
import pandas as pd

def build_concept_to_sents(sentences_path):
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

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_rdm_heatmap(rdm: np.ndarray, title: str, concepts: list = None, save_path: str = None):
    """
    Plots a heatmap of a Representational Dissimilarity Matrix (RDM).

    Args:
        rdm (np.ndarray): The RDM to plot. Expected to be a square matrix.
        title (str): The title of the heatmap plot.
        concepts (list, optional): A list of concept labels corresponding to the RDM rows/columns.
                                   If provided, these will be used as tick labels. Defaults to None.
        save_path (str, optional): Path to save the heatmap image (e.g., "rdm_heatmap.png").
                                   If None, the plot will be displayed. Defaults to None.
    """
    if rdm.ndim != 2 or rdm.shape[0] != rdm.shape[1]:
        raise ValueError("RDM must be a square 2D NumPy array.")

    plt.figure(figsize=(10, 8)) # Adjust figure size as needed for readability
    
    # Use seaborn.heatmap for better aesthetics
    ax = sns.heatmap(
        rdm,
        cmap='viridis', # Or 'magma', 'plasma', 'cividis', 'RdBu_r' for diverging, etc.
        annot=False,    # Set to True to show dissimilarity values on heatmap (can be cluttered for large RDMs)
        fmt=".2f",      # Format for annotations
        square=True,    # Ensure cells are square
        cbar_kws={'label': 'Dissimilarity'} # Label for the color bar
    )

    ax.set_title(title, fontsize=16)
    
    if concepts:
        # Set tick labels if concepts are provided
        ax.set_xticks(np.arange(len(concepts)) + 0.5, minor=False) # Center ticks between cells
        ax.set_yticks(np.arange(len(concepts)) + 0.5, minor=False)
        ax.set_xticklabels(concepts, rotation=90, ha='right', fontsize=8) # Rotate for readability
        ax.set_yticklabels(concepts, rotation=0, va='center', fontsize=8)
        ax.tick_params(axis='x', length=0) # Remove tick marks
        ax.tick_params(axis='y', length=0)
    else:
        # If no concepts, just use numerical indices
        ax.set_xlabel("Concept Index", fontsize=12)
        ax.set_ylabel("Concept Index", fontsize=12)

    plt.tight_layout() # Adjust layout to prevent labels from overlapping

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight') # Save with high resolution
        plt.close() # Close the plot to free memory
        print(f"Heatmap saved to {save_path}")
    else:
        plt.show() # Display the plot

# Example Usage (you would put this in your execute_pipeline.py)
# from rdm_heatmap_plotter import plot_rdm_heatmap # Assuming you save this in a file named rdm_heatmap_plotter.py

# # After computing your brain_group_rdm:
# # plot_rdm_heatmap(brain_group_rdm, "Brain Group RDM", concepts=concepts, save_path="brain_rdm_heatmap.png")

# # Inside your loop for LLM layers:
# # ...
# # if llm_embedding_matrix_for_sents.std() >= 1e-6: # Only plot if RDM is not constant
# #     model_rdm_sents_llm = build_rdm(llm_embedding_matrix_for_sents)
# #     plot_rdm_heatmap(model_rdm_sents_llm, f"LLM Layer {layer_idx} Sentence RDM", concepts=concepts, 
# #                      save_path=f"llm_layer_{layer_idx}_sents_rdm_heatmap.png")
# # ...



