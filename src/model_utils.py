from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from collections import defaultdict

def load_model_and_tokenizer(model_name="bert-base-uncased"):
    # Load model and tokenizer
    model = AutoModel.from_pretrained("bert-base-uncased")
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    return model, tokenizer

def encode_sentences(model, tokenizer, sentence):

    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    
    cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
    return cls_embedding

def encode_word(model, tokenizer, word):
    inputs = tokenizer(word, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze().numpy()




