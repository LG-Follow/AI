# utils/collate.py
import torch
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    # input_ids와 pixel_values가 확실히 Tensor 형식이 되도록 보장
    input_ids = [torch.tensor(item["input_ids"]) if not isinstance(item["input_ids"], torch.Tensor) else item["input_ids"] for item in batch]
    pixel_values = [torch.tensor(item["pixel_values"]) if not isinstance(item["pixel_values"], torch.Tensor) else item["pixel_values"] for item in batch]
    
    # input_ids를 패딩하여 동일한 길이로 맞춤
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=0)
    
    # attention_mask 생성 (패딩된 위치는 0, 그렇지 않은 위치는 1)
    attention_mask = (input_ids_padded != 0).long()

    # attention_mask 확인용 출력 (학습 시에는 삭제)
    print("attention_mask:", attention_mask)

    # pixel_values는 크기가 동일하므로 그대로 stack
    pixel_values_stacked = torch.stack(pixel_values)
    captions = [item["caption"] for item in batch]  # 캡션 리스트

    return input_ids_padded, attention_mask, pixel_values_stacked, captions
