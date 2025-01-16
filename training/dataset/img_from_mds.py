import numpy as np
import random
import streaming
import torch
from streaming import StreamingDataset
from streaming.base.format.mds.encodings import Encoding, _encodings
from typing import Any

class uint8(Encoding):
    def encode(self, obj: Any) -> bytes:
        return obj.tobytes()

    def decode(self, data: bytes) -> Any:
        x=  np.frombuffer(data, np.uint8).astype(np.float32)
        return (x / 255.0 - 0.5) * 24.0

_encodings["uint8"] = uint8

class MdsDataLoader():
    def __init__(self, batch_size, num_workers,
                 image_folder, local_train_dir, 
                 tokenizer, model_max_length, cfg):
        self.tokenizer = tokenizer
        self.model_max_length = model_max_length
        self.cfg = cfg
        
        streaming.base.util.clean_stale_shared_memory()
        train_dataset = StreamingDataset(
            local=local_train_dir,
            remote=image_folder,
            split=None,
            shuffle=True,
            shuffle_algo="naive",
            num_canonical_nodes=1,
            batch_size = batch_size
        )
        
        self.templates = [
            "The image shows a {0}",
            "This is a photo of a {0}",
            "Here’s an image of a {0}",
            "In this image, we see a {0}",
            "Captured here is a {0}",
            "This picture features a {0}",
            "Displayed is a {0}",
            "A picture of a {0}",
            "We have a photo of a {0}",
            "A snapshot of a {0}"
        ]
        
        
        self.dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=self.custom_collate_fn
        )
        self.dataset_length = len(train_dataset)
        
    def custom_collate_fn(self, batch):
        pixel_values = []
        captions = []
    
        for item in batch:
            pixel_value = torch.tensor(item["vae_output"]).reshape(-1, 4, 32, 32)
            pixel_values.append(pixel_value)
            caption = random.choice(self.templates).format(item["label_as_text"].split(',')[0]) if random.random() > self.cfg else ""
            captions.append(caption)
            
        pixel_values = torch.cat(pixel_values).unsqueeze_(2)
        text_tokens_and_mask = self.tokenizer(
            captions,
            max_length=self.model_max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors='pt'
        )
        input_ids = text_tokens_and_mask['input_ids']
        cond_mask = text_tokens_and_mask['attention_mask']

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "cond_mask": cond_mask
        }
    