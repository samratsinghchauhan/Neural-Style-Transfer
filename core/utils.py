from PIL import Image
import numpy as np
import torch

def imcnvt(image):
    x = image.to("cpu").clone().detach().numpy().squeeze()
    x = x.transpose(1,2,0)
    x = x * np.array((0.5, 0.5, 0.5)) + np.array((0.5, 0.5, 0.5))
    return np.clip(x, 0, 1)

def load_image(path, transform, device):
    image = Image.open(path).convert("RGB")
    image = transform(image).to(device)
    return image
