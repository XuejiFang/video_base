from PIL import Image
from diffusers.utils import export_to_video, make_image_grid
import decord
import os
import torch
import torchvision
import numpy as np

def concatenate_videos(all_vid_paths):
    all_videos = []
    for path in all_vid_paths:
        # f h w 3
        video = torchvision.io.read_video(path)[0]
        all_videos.append(video)
    all_videos = torch.stack(all_videos)

    b, f, h, w, c = all_videos.shape
    all_videos = all_videos.cpu().numpy().astype(np.uint8)
    video_grid = []
    for frame_idx in range(f):
        frame_grid = []
        for vid_idx in range(min(b, 16)):
            frame_grid.append(Image.fromarray(all_videos[vid_idx,frame_idx,...]))
        frame_grid = make_image_grid(frame_grid, rows=4, cols=4)
        video_grid.append(frame_grid)

    return video_grid

if __name__ == '__main__':

    # 设置视频文件夹路径
    # video_folder = '/storage/qiguojunLab/fangxueji/Projects/video_base/samples/our_baseline/base-2'
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='/storage/qiguojunLab/fangxueji/Projects/video_base/samples/our_baseline/base-2')
    parser.add_argument('--super_dir', type=str, default=None)
    if parser.parse_args().super_dir is not None:
        super_dir = parser.parse_args().super_dir
        video_folders = [os.path.join(super_dir, f) for f in os.listdir(super_dir) if os.path.isdir(os.path.join(super_dir, f))]
    else:
        video_folders = [parser.parse_args().dir]

    for video_folder in video_folders:
        prompt_cls = video_folder.split('/')[-1]
        video_files = [f for f in os.listdir(video_folder) if f.endswith('.mp4')]
        video_files = sorted(video_files)

        all_vid_paths = [os.path.join(video_folder, vid) for vid in video_files]

        video_grid = concatenate_videos(all_vid_paths)
        export_to_video(video_grid, f'{video_folder}/grid_video_{prompt_cls}.mp4')