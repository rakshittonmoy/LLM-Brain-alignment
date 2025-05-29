import numpy as np
import random
import pandas as pd

SEED = 1
np.random.seed(SEED)
random.seed(SEED)

from pathlib import Path
from load_data_utils import extract_subject_data, load_fmri_file
from brain_utils import get_network_activations, build_rdm
from model_utils import load_model_and_tokenizer
from llm_embeddings import get_llm_embeddings
from process_data_utils import build_concept_to_sents, build_concept_to_images, plot_rdm_heatmap
from evaluate import compute_rsa, sanity_check_rdm
from collections import defaultdict

from vlm_embeddings_resnet import get_visualbert_embeddings
from scipy.spatial.distance import squareform

# === Parameters ===
#participant_ids = [f"P{idx:02d}" for idx in range(1, 2)]

participant_ids=['P01', 'M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07', 'M08', 'M09', 'M10', 'M13', 'M14', 'M15', 'M16', 'M17']
drive_root = Path('./data/Pereira')
network = "languageLH"
sentences_path = Path("./data/screen_sentences.csv")
images_path = "data/image_data/images"
fmri_mat = None
results = []

# === Brain RDMs ===
brain_rdms = []
for pid in participant_ids:
    mat_path = drive_root / pid / "data_180concepts_pictures.mat"
    if not mat_path.exists():
        extract_subject_data(drive_root / f"{pid}.tar", output_dir=drive_root)
    fmri_mat = load_fmri_file(mat_path)
    activations = get_network_activations(fmri_mat, network)
    _, rdm = build_rdm(activations)
    brain_rdms.append(rdm)

brain_group_rdm = np.mean(brain_rdms, axis=0)

print(f"Shape of brain_group_rdm: {brain_group_rdm.shape}")
print(f"Number of dimensions of brain_group_rdm: {brain_group_rdm.ndim}")

# === Load Sentences and images for every concept ===

concepts = [str(cell[0]) for cell in fmri_mat['keyConcept'].squeeze()]

concept_to_sentences = build_concept_to_sents(sentences_path)

concept_to_images = build_concept_to_images(images_path)


# === BERT Model Embeddings ===
word_embeddings_per_concept, sentence_embeddings_per_concept = get_llm_embeddings(concept_to_sentences)


# === Build BERT RDM for words ===
bert_embedding_matrix_for_words = np.vstack([word_embeddings_per_concept[c] for c in concepts])  # shape: (180, D)
print("BERT Embedding matrix shape for words:", bert_embedding_matrix_for_words.shape)

_, model_rdm_words_llm = build_rdm(bert_embedding_matrix_for_words)
print("Model RDM shape for words:", model_rdm_words_llm.shape)

# === Build BERT RDM for sentences ===
bert_embedding_matrix_for_sents = np.vstack([sentence_embeddings_per_concept[c] for c in concepts])  # shape: (180, D)
print("BERT Embedding matrix shape for sentences:", bert_embedding_matrix_for_sents.shape)

_, model_rdm_sents_llm = build_rdm(bert_embedding_matrix_for_sents)
print("Model RDM shape for sentences:", model_rdm_sents_llm.shape)


# Before calling compute_rsa
sanity_check_rdm("Brain", brain_group_rdm)
sanity_check_rdm("Model words", model_rdm_words_llm)
sanity_check_rdm("Model sents", model_rdm_sents_llm)


# === Brain vs LLM with single words ===
corr, pval = compute_rsa(brain_group_rdm, model_rdm_words_llm)
print(f"RSA correlation: {corr:.3f}, p-value: {pval:.5f}")
results.append({"Model": "BERT", "Condition": "Single words", "Correlation": corr, "p-value": pval})

# === Brain vs LLM with sentences ===
corr, pval = compute_rsa(brain_group_rdm, model_rdm_sents_llm)
print(f"RSA correlation: {corr:.3f}, p-value: {pval:.5f}")
results.append({"Model": "BERT", "Condition": "Full sentences", "Correlation": corr, "p-value": pval})

# ========================================================================= #

# === VLM Model Embeddings ===
# word_embeds, sent_embeds = get_visualbert_embeddings(concept_to_sentences, concept_to_images)

# # === Build VLM RDM for words ===
# vlm_embedding_matrix_for_words = np.vstack([word_embeds[c] for c in concepts])  # shape: (180, D)
# print("VLM Embedding matrix shape for words:", vlm_embedding_matrix_for_words.shape)
# _, model_rdm_words_vlm = build_rdm(vlm_embedding_matrix_for_words)
# print("Model RDM shape for words:", model_rdm_words_vlm.shape)

# # === Build VLM RDM for sentences ===
# vlm_embedding_matrix_for_sents = np.vstack([sent_embeds[c] for c in concepts])  # shape: (180, D)
# print("VLM Embedding matrix shape for sentences:", vlm_embedding_matrix_for_sents.shape)
# _, model_rdm_sents_vlm = build_rdm(vlm_embedding_matrix_for_sents)
# print("Model RDM shape for sentences:", model_rdm_sents_vlm.shape)

# # Before calling compute_rsa
# sanity_check_rdm("Brain", brain_group_rdm)
# sanity_check_rdm("Model words", model_rdm_words_vlm)
# sanity_check_rdm("Model sents", model_rdm_sents_vlm)

# # === Brain vs LLM with single words ===
# corr, pval = compute_rsa(brain_group_rdm, model_rdm_words_vlm)
# print(f"RSA correlation: {corr:.3f}, p-value: {pval:.5f}")
# results.append({"Model": "VisualBERT", "Condition": "Single words", "Correlation": corr, "p-value": pval})

# # === Brain vs VLM with sentences ===
# corr, pval = compute_rsa(brain_group_rdm, model_rdm_sents_vlm)
# print(f"RSA correlation: {corr:.3f}, p-value: {pval:.5f}")
# results.append({"Model": "VisualBERT", "Condition": "Full sentences", "Correlation": corr, "p-value": pval})

# # === Save results === #
# df = pd.DataFrame(results)
# df.to_csv(f"rsa_results_{network}.csv", index=False)

## # ================Any layer code========================================================= # #

print("Determining BERT model layers...")
temp_model, _ = load_model_and_tokenizer(model_name="bert-base-uncased")
num_bert_transformer_layers = temp_model.config.num_hidden_layers # Typically 12 for bert-base
# Total outputs are embedding layer (0) + transformer layers (1 to num_bert_transformer_layers)
all_possible_llm_layers = list(range(num_bert_transformer_layers + 1))
del temp_model # Free up memory
print(f"BERT-base-uncased has {len(all_possible_llm_layers)} output layers (0 to {num_bert_transformer_layers}).")


# --- Store results for each LLM layer ---
all_llm_sent_embeds_by_layer = {}
llm_correlation_results_sents = {}
best_llm_sents_correlation = -float('inf')
best_llm_sents_layer = None
best_llm_sents_rdm = None

all_llm_word_embeds_by_layer = {}
llm_correlation_results_words = {}
best_llm_words_correlation = -float('inf')
best_llm_words_layer = None
best_llm_words_rdm = None

print("\n--- Evaluating LLM Embeddings Across Layers ---")
for layer_idx in all_possible_llm_layers:
    print(f"\nProcessing LLM Layer: {layer_idx}")
    
    # Get LLM embeddings for the current layer
    # _ is used because we're currently only interested in sentence embeddings for comparison
    current_llm_word_embeds, current_llm_sent_embeds = get_llm_embeddings(
        concept_to_sentences,
        model_name="bert-base-uncased",
        layer_idx=layer_idx
    )

    all_llm_word_embeds_by_layer[layer_idx] = current_llm_word_embeds
    all_llm_sent_embeds_by_layer[layer_idx] = current_llm_sent_embeds

    # Build LLM RDM for sentences for the current layer
    # Ensure the order of concepts in vstack matches the order used for brain_rdm
    llm_embedding_matrix_for_words = np.vstack([current_llm_word_embeds[c] for c in concepts])
    llm_embedding_matrix_for_sents = np.vstack([current_llm_sent_embeds[c] for c in concepts])
    
    # Words

    # Check for constant input warning before building RDM
    if llm_embedding_matrix_for_words.std() < 1e-6:
        print(f"WARNING: LLM Word Embedding matrix for Layer {layer_idx} has very low standard deviation. RDM might be constant.")
        # If this happens often, you might want to skip correlation for this layer or handle it.
        llm_correlation_results_words[layer_idx] = np.nan # Assign NaN if RDM is expected to be constant
    else:
        model_rdm_full_words_llm, model_rdm_words_llm = build_rdm(llm_embedding_matrix_for_words)

        # Evaluate correlation with brain RDM
        print(f"Evaluating LLM Word RDM for Layer {layer_idx}...")
        current_correlation, pval = compute_rsa(brain_group_rdm, model_rdm_words_llm)
        llm_correlation_results_words[layer_idx] = current_correlation
        print(f"LLM Word RDM (Layer {layer_idx}) vs Brain RDM Correlation: {current_correlation:.4f}")

        if current_correlation > best_llm_words_correlation:
            best_llm_words_correlation = current_correlation
            best_llm_words_layer = layer_idx
            best_llm_words_rdm = model_rdm_full_words_llm

    # Sentences

    # Check for constant input warning before building RDM
    if llm_embedding_matrix_for_sents.std() < 1e-6:
        print(f"WARNING: LLM Sentence Embedding matrix for Layer {layer_idx} has very low standard deviation. RDM might be constant.")
        # If this happens often, you might want to skip correlation for this layer or handle it.
        llm_correlation_results_sents[layer_idx] = np.nan # Assign NaN if RDM is expected to be constant
    else:
        model_rdm_full_sents_llm, model_rdm_sents_llm = build_rdm(llm_embedding_matrix_for_sents)

        # Evaluate correlation with brain RDM
        print(f"Evaluating LLM Sentence RDM for Layer {layer_idx}...")
        current_correlation, pval = compute_rsa(brain_group_rdm, model_rdm_sents_llm)
        llm_correlation_results_sents[layer_idx] = current_correlation
        print(f"LLM Sentence RDM (Layer {layer_idx}) vs Brain RDM Correlation: {current_correlation:.4f}")

        if current_correlation > best_llm_sents_correlation:
            best_llm_sents_correlation = current_correlation
            best_llm_sents_layer = layer_idx
            best_llm_sents_rdm = model_rdm_full_sents_llm

print("\n--- LLM Layer Comparison Results (Words) ---")
for layer_idx in sorted(llm_correlation_results_words.keys()):
    corr = llm_correlation_results_words[layer_idx]
    print(f"Layer {layer_idx}: Correlation = {corr:.4f}")

print(f"\nBest LLM Word Layer: {best_llm_words_layer} with Correlation: {best_llm_words_correlation:.4f}")

print("\n--- LLM Layer Comparison Results (Sentences) ---")
for layer_idx in sorted(llm_correlation_results_sents.keys()):
    corr = llm_correlation_results_sents[layer_idx]
    print(f"Layer {layer_idx}: Correlation = {corr:.4f}")

print(f"\nBest LLM Sentence Layer: {best_llm_sents_layer} with Correlation: {best_llm_sents_correlation:.4f}")


# Plot RDMs (Brain, LLM, VLM)

plot_rdm_heatmap(squareform(brain_group_rdm), "Brain Group RDM", concepts=concepts, save_path="brain_rdm_heatmap.png")
plot_rdm_heatmap(best_llm_words_rdm, "LLM Word RDM", concepts=concepts, save_path="llm_word_rdm_heatmap.png")
plot_rdm_heatmap(best_llm_sents_rdm, "LLM Sentence RDM", concepts=concepts, save_path="llm_sents_rdm_heatmap.png")