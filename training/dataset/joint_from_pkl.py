# Adapted from https://github.com/guoyww/AnimateDiff/blob/main/animatediff/data/dataset.py
import imageio
import numpy as np
import os
import pickle
import random
import torch
import torchvision

from decord import VideoReader
from einops import rearrange
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms

from .utils import CenterCropResizeVideo

def save_videos_grid(videos: torch.Tensor, path: str, rescale=False, n_rows=6, fps=8):
    videos = rearrange(videos, "b c t h w -> t b c h w")
    outputs = []
    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=n_rows)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = (x * 255).numpy().astype(np.uint8)
        outputs.append(x)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimsave(path, outputs, fps=fps)

class Joint_from_PKL(Dataset):
    def __init__(
            self,
            vid_pkl_path, vid_folder,
            img_pkl_path, img_folder,
            sample_size=256, sample_n_frames=16,
            sample_stride=None,
            target_prompt_key="",
            target_path_keys=[],
            tokenizer=None,
            transform=None,
            cfg=None,
            batch_size=None,
            img_ratio=0.2,
            **kwargs,
        ):

        with open(vid_pkl_path, 'rb') as f:
            self.dataset_vid = pickle.load(f)
        self.length_vid = len(self.dataset_vid)

        with open(img_pkl_path, 'rb') as f:
            self.dataset_img = pickle.load(f)
        self.length_img = len(self.dataset_img)
        self.length = self.length_vid

        self.vid_folder         = vid_folder
        self.img_folder         = img_folder
        self.sample_n_frames    = sample_n_frames
        self.sample_stride      = sample_stride
        self.target_prompt_key  = target_prompt_key
        self.target_path_keys   = target_path_keys
        
        self.tokenizer = tokenizer
        self.model_max_length = kwargs.get("max_text_seq_length", 512)
        
        self.sample_size = tuple(sample_size) if not isinstance(sample_size, int) else (sample_size, sample_size)
        self.pixel_transforms = transform
        self.cfg = cfg
        assert self.cfg is not None, "cfg should be set"

        self.batch_size = batch_size
        self.num_trigger = 0
        self.img_ratio = img_ratio
        self.use_img = False if random.random() > img_ratio else True
    
    def get_batch(self, idx):
        # import pdb; pdb.set_trace()
        if self.num_trigger % self.batch_size == 0:
            flag = random.random()
            self.use_img = True if flag <= self.img_ratio else False
            # print("Reset use_img:", flag,  self.use_img)
        
        if self.use_img:
            idx_img = random.randint(0, self.length_img-1)
            img_dict = self.dataset_img[idx_img]
            img_path = img_dict["path"]
            text_img = img_dict[self.target_prompt_key] if random.random() > self.cfg else ""
            num_imgs = 1
            # TODO: text_preprocessing?
            text = text_img
            pixel_values_img = torchvision.io.read_image(os.path.join(self.img_folder, img_path), mode=torchvision.io.image.ImageReadMode.RGB).unsqueeze(1).contiguous()
            pixel_values_img = pixel_values_img / 255.
            pixel_values_img = self.pixel_transforms(pixel_values_img)

            pixel_values = pixel_values_img
        else:
            video_dict = self.dataset_vid[idx]
            text_vid = video_dict[self.target_prompt_key] if random.random() > self.cfg else ""
            num_imgs = 0
            text = text_vid

            video_path = video_dict[self.target_path_keys[0]]
            video_reader = VideoReader(video_path)

            video_length = len(video_reader)

            if self.sample_n_frames > video_length:
                raise ValueError("sample_n_frames cannot be greater than video_length")
            
            clip_length = min(video_length, (self.sample_n_frames - 1) * self.sample_stride + 1)
            start_idx   = random.randint(0, video_length - clip_length)
            batch_index = np.linspace(start_idx, start_idx + clip_length - 1, self.sample_n_frames, dtype=int)

            # F H W C -> C F H W 0~1
            pixel_values = torch.from_numpy(video_reader.get_batch(batch_index).asnumpy()).permute(3, 0, 1, 2).contiguous()
            pixel_values = pixel_values / 255.
            del video_reader
            pixel_values = self.pixel_transforms(pixel_values)


        text_tokens_and_mask = self.tokenizer(
            text,
            max_length=self.model_max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors='pt'
        )
        input_ids = text_tokens_and_mask['input_ids']
        cond_mask = text_tokens_and_mask['attention_mask']
    

        return dict(pixel_values=pixel_values, input_ids=input_ids, cond_mask=cond_mask, num_imgs=num_imgs)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        
        try:
            sample = self.get_batch(idx)
            self.num_trigger += 1
            return sample
        except Exception as e:
            print(f"Error in getting sample {idx}: {e}")
            idx = random.randint(0, self.length-1)
            print(f"Retry with new index {idx}/{self.length}")
            return self.__getitem__(idx)

    
class JointDataLoader():
    def __init__(
            self, 
            tokenizer, cfg, max_text_seq_length,
            vid_pkl_path, vid_folder,
            img_pkl_path, img_folder,
            target_path_keys, target_prompt_key,
            max_height, max_width, num_frames, sample_rate,
            batch_size, num_workers, img_ratio
    ):
        resize = [CenterCropResizeVideo((max_height, max_width)), ]
        transform = transforms.Compose([
            # ToTensorVideo(),
            *resize, 
            transforms.Lambda(lambda x: 2. * x - 1.)
        ])        

        train_dataset = Joint_from_PKL(
            vid_pkl_path=vid_pkl_path,
            vid_folder=vid_folder,
            img_pkl_path=img_pkl_path,
            img_folder=img_folder,
            sample_size=(max_height, max_width),
            sample_n_frames=num_frames,
            sample_stride=sample_rate,
            target_path_keys=target_path_keys.split(','),
            target_prompt_key=target_prompt_key,
            tokenizer=tokenizer,
            transform=transform,
            cfg=cfg,
            batch_size=batch_size,
            max_text_seq_length=max_text_seq_length,
            img_ratio=img_ratio
        )

        self.dataloader = DataLoader(
            train_dataset,
            shuffle=True,
            pin_memory=True,
            batch_size=batch_size,
            num_workers=num_workers,
            drop_last=True,
        )

        self.dataset_length = len(train_dataset)

if __name__ == "__main__":
    pass
    # from einops import rearrange
    # import imageio
    # import numpy as np
    # import os
    # import torchvision
    # def save_videos_grid(videos: torch.Tensor, path: str, rescale=False, n_rows=6, fps=8):
    #     videos = rearrange(videos, "b c t h w -> t b c h w")
    #     outputs = []
    #     for x in videos:
    #         x = torchvision.utils.make_grid(x, nrow=n_rows)
    #         x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
    #         if rescale:
    #             x = (x + 1.0) / 2.0  # -1,1 -> 0,1
    #         x = (x * 255).cpu().numpy().astype(np.uint8)
    #         outputs.append(x)

    #     os.makedirs(os.path.dirname(path), exist_ok=True)
    #     imageio.mimsave(path, outputs, fps=fps)
    # save_dir = './tmp'
    # prompts = ""
    # for idx, batch in enumerate(dataloader):
    #     # pixel_values: BCFHW, input_ids/cond_mask: Bx1x512, 
    #     print(batch.keys(), batch["pixel_values"].min(), batch["pixel_values"].max())
    #     print(batch["pixel_values"].shape, batch["input_ids"].shape, batch["cond_mask"].shape, batch["num_imgs"])
    #     for i in range(batch["pixel_values"].shape[0]):
    #         # -1 ~ 1 BCFHW
    #         save_videos_grid(batch["pixel_values"][i:i+1], os.path.join(save_dir, f"{idx}-{i}.mp4"), rescale=True)
    #         prompts += f"{idx}-{i}\t: {batch['input_ids'][i]}\n"
    #     if idx == 3:
    #         break
    # with open(os.path.join(save_dir, 'prompts.txt'), 'w') as f:
    #     f.write(prompts)