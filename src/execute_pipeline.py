from pathlib import Path
from load_data import extract_subject_data, load_fmri_file
from brain_utils import get_network_activations, build_rdm
from model_utils import load_model_and_tokenizer
from llm_embeddings import get_llm_embeddings
from evaluate import compute_rsa, sanity_check_rdm
from collections import defaultdict
import numpy as np
import pandas as pd
import random

# === Parameters ===
participant_ids = [f"P{idx:02d}" for idx in range(1, 2)]

# participant_ids=['P01', 'M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07', 'M08', 'M09', 'M10', 'M13', 'M14', 'M15', 'M16', 'M17']
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
    print("RDM shape", rdm.shape)
    brain_rdms.append(rdm)

brain_group_rdm = np.mean(brain_rdms, axis=0)

# === Load Sentences ===
df = pd.read_csv(sentences_path)

concepts = [str(cell[0]) for cell in fmri_mat['keyConcept'].squeeze()]

concept_to_sentences = defaultdict(list)

for _, row in df.iterrows():
    concept = row['Concept'].lower()
    sentence = row['Sentence_Screen']
    concept_to_sentences[concept].append(sentence)


# === Model Embeddings ===
word_embeddings_per_concept, sentence_embeddings_per_concept = get_llm_embeddings(concept_to_sentences)

# === Build Model RDM for words ===
bert_embedding_matrix_for_words = np.vstack([word_embeddings_per_concept[c] for c in concepts])  # shape: (180, D)
print("BERT Embedding matrix shape for words:", bert_embedding_matrix_for_words.shape)
model_rdm_words = build_rdm(bert_embedding_matrix_for_words)
print("Model RDM shape for words:", model_rdm_words.shape)

# === Build Model RDM for sentences ===
bert_embedding_matrix_for_sents = np.vstack([sentence_embeddings_per_concept[c] for c in concepts])  # shape: (180, D)
print("BERT Embedding matrix shape for sentences:", bert_embedding_matrix_for_sents.shape)
model_rdm_sents = build_rdm(bert_embedding_matrix_for_sents)
print("Model RDM shape for sentences:", model_rdm_sents.shape)


# Before calling compute_rsa
sanity_check_rdm("Brain", brain_group_rdm)
sanity_check_rdm("Model words", model_rdm_words)
sanity_check_rdm("Model sents", model_rdm_sents)


# === Brain vs LLM with single words ===
corr = compute_rsa(brain_group_rdm, model_rdm_words)
print(f"RSA correlation (brain vs. model): {corr}")

# === Brain vs VLM with sentences ===
corr = compute_rsa(brain_group_rdm, model_rdm_sents)
print(f"RSA correlation (brain vs. model): {corr}")

