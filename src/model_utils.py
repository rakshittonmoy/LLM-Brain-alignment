# from transformers import AutoTokenizer, AutoModel
# import torch
# import numpy as np
# from collections import defaultdict

# def load_model_and_tokenizer(model_name="bert-base-uncased"):
#     # Load model and tokenizer
#     model = AutoModel.from_pretrained("bert-base-uncased")
#     model.eval()
#     tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
#     return model, tokenizer

# def encode_sentences(model, tokenizer, sentence):

#     inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True)
#     with torch.no_grad():
#         outputs = model(**inputs)
    
#     cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
#     return cls_embedding

# def encode_word(model, tokenizer, word):
#     inputs = tokenizer(word, return_tensors='pt')
#     with torch.no_grad():
#         outputs = model(**inputs)
#     return outputs.last_hidden_state[:, 0, :].squeeze().numpy()


import torch
from transformers import AutoModel, AutoTokenizer
import numpy as np

# Define the device globally
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model_and_tokenizer(model_name="bert-base-uncased"):
    """
    Loads a pre-trained BERT model and tokenizer.
    The model is configured to output all hidden states.

    Args:
        model_name (str): The name of the pre-trained BERT model to load.

    Returns:
        tuple: A tuple containing the loaded model and tokenizer.
    """
    # MODIFIED: Add output_hidden_states=True to get all layer outputs
    model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
    model.eval() # Set the model to evaluation mode
    model.to(device) # Move the model to the appropriate device (GPU/CPU)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def encode_sentences(model, tokenizer, sentence, layer_idx: int = -1):
    """
    Encodes a sentence using the provided BERT model and tokenizer,
    extracting the CLS token embedding from a specified layer.

    Args:
        model (PreTrainedModel): The loaded BERT model.
        tokenizer (PreTrainedTokenizer): The loaded BERT tokenizer.
        sentence (str): The input sentence to encode.
        layer_idx (int): The index of the hidden layer to extract embeddings from.
                         -1 (default) for the last layer.
                         0 for the embedding layer output.
                         1 for the first transformer layer output, and so on.

    Returns:
        numpy.ndarray: The CLS token embedding from the specified layer.
    """
    # Tokenize the input sentence and move inputs to the correct device
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True).to(device)

    with torch.no_grad():
        # Perform a forward pass through the model
        outputs = model(**inputs)
    
    # Access the hidden states from the specified layer
    # outputs.hidden_states is a tuple where:
    # outputs.hidden_states[0] is the embedding layer output
    # outputs.hidden_states[1] is the output of the first transformer layer
    # ...
    # outputs.hidden_states[N] is the output of the N-th transformer layer
    selected_layer_hidden_state = outputs.hidden_states[layer_idx]

    # Extract the CLS token (first token) embedding from the selected layer
    # .squeeze() removes dimensions of size 1 (e.g., from [1, 1, D] to [D])
    # .cpu().numpy() moves the tensor to CPU and converts it to a NumPy array
    cls_embedding = selected_layer_hidden_state[:, 0, :].squeeze().cpu().numpy()
    return cls_embedding

def encode_word(model, tokenizer, word, layer_idx: int = -1):
    """
    Encodes a single word using the provided BERT model and tokenizer,
    extracting the CLS token embedding from a specified layer.

    Args:
        model (PreTrainedModel): The loaded BERT model.
        tokenizer (PreTrainedTokenizer): The loaded BERT tokenizer.
        word (str): The input word to encode.
        layer_idx (int): The index of the hidden layer to extract embeddings from.
                         -1 (default) for the last layer.
                         0 for the embedding layer output.
                         1 for the first transformer layer output, and so on.

    Returns:
        numpy.ndarray: The CLS token embedding from the specified layer.
    """
    # Tokenize the input word and move inputs to the correct device
    inputs = tokenizer(word, return_tensors='pt').to(device)

    with torch.no_grad():
        # Perform a forward pass through the model
        outputs = model(**inputs)
    
    # Access the hidden states from the specified layer
    selected_layer_hidden_state = outputs.hidden_states[layer_idx]

    # Extract the CLS token (first token) embedding from the selected layer
    cls_embedding = selected_layer_hidden_state[:, 0, :].squeeze().cpu().numpy()
    return cls_embedding





