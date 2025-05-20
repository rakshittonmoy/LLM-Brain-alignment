from pathlib import Path
from load_data import extract_subject_data, load_fmri_file
from brain_utils import get_network_activations, build_rdm
from model_utils import load_model_and_tokenizer, encode_sentences
# from evaluate import compute_rsa
import numpy as np
import pandas as pd
import random

# === Parameters ===
participant_ids = [f"P{idx:02d}" for idx in range(1, 2)]

participant_ids=['P01', 'M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07', 'M08', 'M09', 'M10', 'M13', 'M14', 'M15', 'M16', 'M17']
drive_root = Path('./data/Pereira') 
network = "languageLH"
SEED = 123
np.random.seed(SEED)
random.seed(SEED)
sentences_path = Path("./data/screen_sentences.csv")
column_name = "Sentence_Screen"
fmri_mat = None

# === Brain RDMs ===
brain_rdms = []
for pid in participant_ids:
    mat_path = drive_root / pid / "data_180concepts_pictures.mat"
    if not mat_path.exists():
        extract_subject_data(drive_root / f"{pid}.tar", output_dir=drive_root)
    fmri_mat = load_fmri_file(mat_path)
    activations = get_network_activations(fmri_mat, network)
    rdm = build_rdm(activations)
    brain_rdms.append(rdm)


# === Load Sentences ===
from collections import defaultdict

df = pd.read_csv(sentences_path)

concept_to_sentences = defaultdict(list)
concept_embeddings = {}

# === Model Embeddings ===

model, tokenizer = load_model_and_tokenizer("bert-base-uncased")


concepts = [str(cell[0]) for cell in fmri_mat['keyConcept'].squeeze()]
#print("Concepts:", concepts)

for _, row in df.iterrows():
    concept = row['Concept']
    sentence = row['Sentence_Screen']
    concept_to_sentences[concept].append(sentence)

print("Concepts in sentences:", concept_to_sentences)

for c, sents in concept_to_sentences.items():
    if len(sents) < 5:
        print(f"{c} has {len(sents)} sentences!")

    sent_embeddings = []

    for sentence in sents:
        cls_embedding = encode_sentences(model, tokenizer, sentence)
        sent_embeddings.append(cls_embedding)

    concept_embeddings[c.lower()] = np.mean(sent_embeddings, axis=0)

print("Concept embeddings shape:", len(concept_embeddings), concept_embeddings['ability'].shape)

# === Build Model RDM ===
embedding_matrix = np.vstack([concept_embeddings[c] for c in concepts])  # shape: (180, D)
print("Embedding matrix shape:", embedding_matrix.shape)

model_rdm = build_rdm(embedding_matrix)


# === Final Evaluation ===
brain_group_rdm = np.mean(brain_rdms, axis=0)
#corr = compute_rsa(brain_group_rdm, model_rdm)
corr = None

#print(f"RSA correlation (brain vs. model): {corr:.4f}")
