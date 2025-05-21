import numpy as np
from model_utils import load_model_and_tokenizer, encode_word, encode_sentences

def get_llm_embeddings(concept_to_sentences, model_name="bert-base-uncased"):

    model, tokenizer = load_model_and_tokenizer(model_name)
    word_embeddings_per_concept = {}
    sentence_embeddings_per_concept = {}

    for concept, sents in concept_to_sentences.items():
        if len(sents) < 5:
            print(f"{concept} has {len(sents)} sentences!")

        word_embeddings_per_concept[concept] = encode_word(model, tokenizer, concept)

        sent_embeddings = []

        for sentence in sents:
            cls_embedding = encode_sentences(model, tokenizer, sentence)
            sent_embeddings.append(cls_embedding)

        sentence_embeddings_per_concept[concept] = np.mean(sent_embeddings, axis=0)

    print("Concept embeddings shape:", len(sentence_embeddings_per_concept), sentence_embeddings_per_concept['ability'].shape)

    return word_embeddings_per_concept, sentence_embeddings_per_concept
  
