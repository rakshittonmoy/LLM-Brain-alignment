# import numpy as np
# from model_utils import load_model_and_tokenizer, encode_word, encode_sentences

# def get_llm_embeddings(concept_to_sentences, model_name="bert-base-uncased"):

#     model, tokenizer = load_model_and_tokenizer(model_name)
#     word_embeddings_per_concept = {}
#     sentence_embeddings_per_concept = {}

#     for concept, sents in concept_to_sentences.items():
#         if len(sents) < 5:
#             print(f"{concept} has {len(sents)} sentences!")

#         word_embeddings_per_concept[concept] = encode_word(model, tokenizer, concept)

#         sent_embeddings = []

#         for sentence in sents:
#             cls_embedding = encode_sentences(model, tokenizer, sentence)
#             sent_embeddings.append(cls_embedding)

#         sentence_embeddings_per_concept[concept] = np.mean(sent_embeddings, axis=0)

#     print("Concept embeddings shape:", len(sentence_embeddings_per_concept), sentence_embeddings_per_concept['ability'].shape)

#     return word_embeddings_per_concept, sentence_embeddings_per_concept

import numpy as np
# Assuming model_utils.py contains the modified load_model_and_tokenizer, encode_word, encode_sentences
from model_utils import load_model_and_tokenizer, encode_word, encode_sentences

# MODIFIED: Added layer_idx parameter with a default of -1 (last layer)
def get_llm_embeddings(concept_to_sentences, model_name="bert-base-uncased", layer_idx: int = 12):

    model, tokenizer = load_model_and_tokenizer(model_name)
    word_embeddings_per_concept = {}
    sentence_embeddings_per_concept = {}

    for concept, sents in concept_to_sentences.items():
        if len(sents) < 5:
            print(f"{concept} has {len(sents)} sentences!")

        # MODIFIED: Pass layer_idx to encode_word
        word_embeddings_per_concept[concept] = encode_word(model, tokenizer, concept, layer_idx=layer_idx)

        sent_embeddings = []

        for sentence in sents:
            # MODIFIED: Pass layer_idx to encode_sentences
            cls_embedding = encode_sentences(model, tokenizer, sentence, layer_idx=layer_idx)
            sent_embeddings.append(cls_embedding)

        sentence_embeddings_per_concept[concept] = np.mean(sent_embeddings, axis=0)

    print("Concept embeddings shape:", len(sentence_embeddings_per_concept), sentence_embeddings_per_concept['ability'].shape)

    return word_embeddings_per_concept, sentence_embeddings_per_concept

  
