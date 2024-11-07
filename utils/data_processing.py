# utils/data_processing.py
# utils/data_processing.py
from datasets import load_dataset
from transformers import BlipProcessor
import torch

def load_flickr30k_data():
    # Flickr30k 데이터셋 로드
    dataset = load_dataset("nlphuji/flickr30k", split="test")
    print("Dataset loaded:", dataset)
    return dataset

def preprocess_data(example, processor):
    # 이미지 객체와 캡션을 전처리
    image = example["image"]  # PIL 이미지 객체
    inputs = processor(images=image, text=example["caption"][0], return_tensors="pt", padding=True)

    # inputs["input_ids"]와 inputs["pixel_values"]가 Tensor가 아닌 경우 강제 변환 후 squeeze
    input_ids = inputs["input_ids"]
    pixel_values = inputs["pixel_values"]

    if not isinstance(input_ids, torch.Tensor):
        input_ids = torch.tensor(input_ids)
    if not isinstance(pixel_values, torch.Tensor):
        pixel_values = torch.tensor(pixel_values)

    # 배치 차원 제거
    input_ids = input_ids.squeeze(0)
    pixel_values = pixel_values.squeeze(0)

    # dict 형태로 반환
    return {
        "input_ids": input_ids,
        "pixel_values": pixel_values,
        "caption": example["caption"][0]
    }