from transformers import CLIPProcessor, CLIPModel
import torch
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_clip_model():
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    return model, processor

def generate_clip_prompt(model, processor, image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)

    candidate_texts = ["a sunny day", "a person smiling", "a cityscape", "a painting of a landscape",
                       "a dark forest", "a beautiful sunset", "an abstract artwork", "a crowded street"]

    text_inputs = processor(text=candidate_texts, return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
        text_features = model.get_text_features(**text_inputs)

    similarity = torch.matmul(image_features, text_features.T)
    
    most_similar_index = similarity.argmax().item()
    prompt = candidate_texts[most_similar_index]

    return prompt
