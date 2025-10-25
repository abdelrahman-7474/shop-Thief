# utils.py
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

# same transform you trained with
frame_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.0),
    transforms.ToTensor()
])

def sample_16_frames(cap):
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if total <= 0:
        return []

    # pick 16 indices
    if total >= 16:
        indices = np.linspace(0, total - 1, 16, dtype=int)
    else:
        indices = np.linspace(0, total - 1, total, dtype=int)
        # pad by repeating last index
        pad = np.full(16 - len(indices), indices[-1], dtype=int)
        indices = np.concatenate([indices, pad])

    frames = []
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ret, frame = cap.read()
        if not ret or frame is None:

            if frames:
                frames.append(frames[-1])
            else:
                frames.append(torch.zeros(3, 224, 224))
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(frame)
        tensor = frame_transform(pil)  # C,H,W
        frames.append(tensor)
    return frames  # list of 16 tensors (C,H,W)

def video_to_tensor_3d(video_path: str, device: str = "cpu"):
    cap = cv2.VideoCapture(video_path)
    try:
        frames = sample_16_frames(cap)
    finally:
        cap.release()

    if len(frames) == 0:
        raise ValueError("Could not read frames from video.")

    clip = torch.stack(frames)             # (16, C, H, W)
    clip = clip.permute(1, 0, 2, 3)        # (C, T, H, W)
    clip = clip.unsqueeze(0).to(device)    # (1, C, T, H, W)
    return clip