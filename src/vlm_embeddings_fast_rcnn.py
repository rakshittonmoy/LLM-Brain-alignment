import torch
import numpy as np
from PIL import Image
from transformers import VisualBertModel, BertTokenizerFast, VisualBertForQuestionAnswering
from torchvision import transforms

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.structures import ImageList

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


import torch
import numpy as np
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.structures import ImageList
from torchvision.transforms.functional import to_pil_image

@torch.no_grad()
def extract_visual_features(image_tensor):
    # Convert normalized tensor back to PIL for Detectron2
    image = to_pil_image(image_tensor)

    # Detectron2 config
    cfg = get_cfg()
    #cfg.merge_from_file("detectron2/configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_R_101_FPN_3x/137849458/model_final_f6e8b1.pkl"
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_model(cfg)
    model.eval()
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)

    # Prepare image for model
    transform = transforms.Compose([
        transforms.Resize((800, 800)),  # Optional: resize for consistency
        transforms.ToTensor()
    ])
    image_tensor = transform(image).to(cfg.MODEL.DEVICE)
    images = ImageList.from_tensors([image_tensor], model.backbone.size_divisibility)

    # Run model
    features = model.backbone(images.tensor)
    proposals, _ = model.proposal_generator(images, features, None)
    box_features = model.roi_heads._shared_roi_transform(
        [features[f] for f in model.roi_heads.in_features], proposals[0].proposal_boxes
    )

    # Project to 2048-dim
    pooled = model.roi_heads.box_head(box_features)  # [num_regions, 1024]
    visual_embeds = model.roi_heads.box_predictor.cls_score(pooled)  # [num_regions, num_classes]

    # Alternatively: take pooled before prediction
    # Resize or select top-N (e.g., 36 regions)
    pooled = pooled[:36]
    return pooled  # shape: [36, 1024] or whatever final dim is

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
    #visual_embeds = image_tensor.unsqueeze(0).to(device)  # [1, 3, 224, 224]
    visual_embeds = extract_visual_features(image_tensor).unsqueeze(0).to(device)

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

    print("vlm embeddings", word_embeddings_per_concept, sentence_embeddings_per_concept)

    return word_embeddings_per_concept, sentence_embeddings_per_concept
