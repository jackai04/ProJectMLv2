# ProJectML

วิธีเรียกใช้ model

import torch
from model import BetterCNN  # ถ้ามีไฟล์ model.py แยกไว้

model = BetterCNN()
checkpoint = torch.load("CIFAR10_model.pth", map_location="cpu")
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
