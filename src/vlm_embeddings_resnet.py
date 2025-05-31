import torch
import numpy as np
from PIL import Image
from transformers import VisualBertModel, BertTokenizerFast, VisualBertForQuestionAnswering
from torchvision import transforms
import torchvision.models as models # ADDED: Import torchvision models

# Removed: detectron2 imports are no longer needed for visual feature extraction
# from detectron2 import model_zoo
# from detectron2.config import get_cfg
# from detectron2.modeling import build_model
# from detectron2.checkpoint import DetectionCheckpointer
# from detectron2.structures import ImageList
# from torchvision.transforms.functional import to_pil_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the ResNet model globally to avoid re-loading it for every image
# This model will be loaded once when the script starts
_resnet_model = None

@torch.no_grad()
def extract_visual_features(image_tensor):
    global _resnet_model # Declare intent to modify global variable

    if _resnet_model is None:
        # Load a pre-trained ResNet-101 model
        _resnet_model = models.resnet101(pretrained=True)
        # Remove the final classification layer to get features
        _resnet_model = torch.nn.Sequential(*list(_resnet_model.children())[:-1])
        _resnet_model.to(device)
        _resnet_model.eval() # Set to evaluation mode

    # The input image_tensor is already preprocessed (resized, normalized, ToTensor)
    # It should be [C, H, W] for a single image, or [B, C, H, W] for a batch
    # Unsqueeze if it's a single image [C, H, W] -> [1, C, H, W]
    if image_tensor.ndim == 3:
        image_tensor = image_tensor.unsqueeze(0)

    # Ensure the image tensor is on the correct device
    image_tensor = image_tensor.to(device)

    # Extract features
    features = _resnet_model(image_tensor)

    # Reshape features to [batch_size, feature_dim]
    # For ResNet after removing the last layer, it's typically [1, 2048, 1, 1]
    # We want [1, 2048] for VisualBERT's visual_embeds
    features = features.view(features.size(0), -1) # Flatten spatial dimensions

    # VisualBERT expects a sequence of visual features.
    # A common practice is to treat the global pooled feature as a single "region"
    # or to expand it to match the expected number of regions (e.g., 36) if VisualBERT
    # expects a fixed number. For simplicity and initial testing, we'll use a single
    # pooled feature. If VisualBERT requires 36 regions, we'd need to replicate this.
    # VisualBERT's visual_embeds expects shape (batch_size, num_visual_tokens, visual_embed_dim)
    # ResNet101 outputs 2048 features. So, this will be (1, 1, 2048).
    # The original Detectron2 extracted 36 regions, each 1024-dim.
    # We will return a single feature vector for now.
    # If VisualBERT expects a sequence, we might need to unsqueeze(1) later.
    # For now, let's return [1, 2048] as a single "region".
    return features.squeeze(0) # Returns [2048] for a single image


def load_visualbert():
    # VisualBERT expects visual_embeds of shape (batch_size, num_visual_tokens, visual_embed_dim)
    # The default VisualBERT model expects visual_embeds of size 2048, and typically 36 tokens.
    # Since our ResNet extracts a single 2048-dim feature, we will adjust the VisualBERT model
    # to expect 1 visual token. This is a common pattern for simpler VLM integrations.
    # If this causes issues, we might need to pad/replicate the ResNet features to 36 tokens.

    # We will load the model and then potentially modify its visual input layer if needed.
    # However, VisualBERT's `visual_embeds` parameter is often flexible.
    model = VisualBertModel.from_pretrained("uclanlp/visualbert-vqa-coco-pre", output_hidden_states=True).to(device)
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    # Check if VisualBERT's visual_embedding layer needs adjustment
    # The default VisualBERT-VQA-COCO-Pre model expects 2048-dim visual features,
    # which matches our ResNet output. It also expects `num_visual_tokens` which is 36 by default.
    # If VisualBERT throws an error about shape mismatch for `visual_embeds`, we might need to
    # replicate the ResNet feature 36 times. For now, let's assume it can handle 1 token.
    # The `visual_embeds` input to VisualBERT is (batch_size, num_visual_tokens, visual_embed_dim).
    # Our `extract_visual_features` returns (visual_embed_dim,) for a single image.
    # In `encode_image_text`, we unsqueeze it to (1, visual_embed_dim).
    # We need to unsqueeze again to (1, 1, visual_embed_dim) to match VisualBERT's expected (B, N, D)
    # This is handled in `encode_image_text`.

    return model, tokenizer

def preprocess_image(image_path):
    # ResNet typically expects 224x224 input
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),  # Converts to [C, H, W]
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image)

def encode_image_text(model, tokenizer, text, image_tensor, layer_idx: int = -1):
    # Simulate VisualBERT-compatible input
    inputs = tokenizer(text, return_tensors="pt").to(device)

    # extract_visual_features now returns [2048] for a single image
    # We need to reshape it to [1, 1, 2048] for VisualBERT: (batch_size, num_visual_tokens, visual_embed_dim)
    visual_embeds = extract_visual_features(image_tensor).unsqueeze(0).unsqueeze(0).to(device) # [1, 1, 2048]

    # VisualBERT uses these to indicate the type and attention for visual tokens.
    # Since we have 1 visual token, these masks/ids should reflect that.
    visual_token_type_ids = torch.zeros((1, 1), dtype=torch.long).to(device) # Usually 0 for text, 1 for visual
    visual_attention_mask = torch.ones((1, 1), dtype=torch.long).to(device)

    outputs = model(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        token_type_ids=inputs["token_type_ids"],
        visual_embeds=visual_embeds,
        visual_token_type_ids=visual_token_type_ids,
        visual_attention_mask=visual_attention_mask,
    )

    selected_layer_hidden_state = outputs.hidden_states[layer_idx]

    # Extract the CLS token (first token) embedding from the selected layer
    cls_embedding = selected_layer_hidden_state[:, 0, :].squeeze().cpu().detach().numpy()

    return cls_embedding
    #return outputs.last_hidden_state[:, 0, :].squeeze().cpu().detach().numpy()  # CLS token

def get_visualbert_embeddings(concept_to_sentences, concept_to_images, layer_idx: int = 12):
    model, tokenizer = load_visualbert()

    word_embeddings_per_concept = {}
    sentence_embeddings_per_concept = {}

    for concept, sentences in concept_to_sentences.items():
        image_paths = concept_to_images[concept]  # Should be 6 images
        word_embeds = []
        for img_path in image_paths:
            image_tensor = preprocess_image(img_path)
            embed = encode_image_text(model, tokenizer, concept, image_tensor, layer_idx=layer_idx)
            word_embeds.append(embed)
        word_embeddings_per_concept[concept] = np.mean(word_embeds, axis=0)

        sentence_embeds = []
        for sentence in sentences:
            for img_path in image_paths:
                image_tensor = preprocess_image(img_path)
                embed = encode_image_text(model, tokenizer, sentence, image_tensor, layer_idx=layer_idx)
                sentence_embeds.append(embed)
        sentence_embeddings_per_concept[concept] = np.mean(sentence_embeds, axis=0)


    return word_embeddings_per_concept, sentence_embeddings_per_concept
