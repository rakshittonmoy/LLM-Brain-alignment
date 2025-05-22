import torch
import numpy as np
from PIL import Image
from transformers import VisualBertModel, BertTokenizerFast, VisualBertForQuestionAnswering
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_visualbert():
    model = VisualBertModel.from_pretrained("uclanlp/visualbert-vqa-coco-pre").to(device)
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    return model, tokenizer

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),  # Converts to [C, H, W]
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image)

def encode_image_text(model, tokenizer, text, image_tensor):
    # Simulate VisualBERT-compatible input
    inputs = tokenizer(text, return_tensors="pt").to(device)
    visual_embeds = image_tensor.unsqueeze(0).to(device)  # [1, 3, 224, 224]

    # Fake visual token type (VisualBERT doesn't use actual vision transformer)
    visual_token_type_ids = torch.ones((1, 1), dtype=torch.long).to(device)
    visual_attention_mask = torch.ones((1, 1), dtype=torch.long).to(device)

    outputs = model(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        token_type_ids=inputs["token_type_ids"],
        visual_embeds=visual_embeds,
        visual_token_type_ids=visual_token_type_ids,
        visual_attention_mask=visual_attention_mask,
    )
    return outputs.last_hidden_state[:, 0, :].squeeze().cpu().detach().numpy()  # CLS token

def get_visualbert_embeddings(concept_to_sentences, concept_to_images):
    model, tokenizer = load_visualbert()

    word_embeddings_per_concept = {}
    sentence_embeddings_per_concept = {}

    for concept, sentences in concept_to_sentences.items():
        image_paths = concept_to_images[concept]  # Should be 6 images

        word_embeds = []
        for img_path in image_paths:
            image_tensor = preprocess_image(img_path)
            embed = encode_image_text(model, tokenizer, concept, image_tensor)
            word_embeds.append(embed)
        word_embeddings_per_concept[concept] = np.mean(word_embeds, axis=0)

        sentence_embeds = []
        for sentence in sentences:
            for img_path in image_paths:
                image_tensor = preprocess_image(img_path)
                embed = encode_image_text(model, tokenizer, sentence, image_tensor)
                sentence_embeds.append(embed)
        sentence_embeddings_per_concept[concept] = np.mean(sentence_embeds, axis=0)

    return word_embeddings_per_concept, sentence_embeddings_per_concept
