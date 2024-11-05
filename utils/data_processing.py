import os
from PIL import Image

def load_image(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
        return image
    except Exception as e:
        print(f"Error loading image at {image_path}: {e}")
        return None

def get_image_paths(data_dir):
    supported_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    return [os.path.join(data_dir, file) for file in os.listdir(data_dir) if file.lower().endswith(supported_extensions)]
