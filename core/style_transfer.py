import torch
from torchvision import models
from core.utils import imcnvt, load_image
import matplotlib.pyplot as plt
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

transform = torch.nn.Sequential(
    torch.nn.Upsample(size=(300, 300), mode='bilinear'),
    torch.nn.BatchNorm2d(3)
)

transform_fn = torch.nn.Sequential(
    torch.nn.Identity()
)

vgg = models.vgg19(pretrained=True).features.to(device).eval()
for param in vgg.parameters():
    param.requires_grad = False

def get_features(image, model):
    layers = {
        '0': 'conv1_1', '5': 'conv2_1', '10': 'conv3_1',
        '19': 'conv4_1', '21': 'conv4_2', '28': 'conv5_1'
    }
    features = {}
    x = image.unsqueeze(0)
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
    return features

def gram_matrix(tensor):
    _, d, h, w = tensor.size()
    tensor = tensor.view(d, h * w)
    gram = torch.mm(tensor, tensor.t())
    return gram

def stylize(content_path, style_path, output_dir='outputs', epochs=500, print_after=500):
    from torchvision import transforms

    os.makedirs(output_dir, exist_ok=True)

    preprocess = transforms.Compose([
        transforms.Resize(300),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    content = load_image(content_path, preprocess, device)
    style = load_image(style_path, preprocess, device)
    target = content.clone().requires_grad_(True)

    style_weights = {
        'conv1_1': 1.0, 'conv2_1': 0.8,
        'conv3_1': 0.4, 'conv4_1': 0.2,
        'conv5_1': 0.1
    }

    content_weight = 100
    style_weight = 1e8

    content_features = get_features(content, vgg)
    style_features = get_features(style, vgg)
    style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

    optimizer = torch.optim.Adam([target], lr=0.007)

    for i in range(1, epochs + 1):
        target_features = get_features(target, vgg)

        content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2']) ** 2)

        style_loss = 0
        for layer in style_weights:
            target_gram = gram_matrix(target_features[layer])
            style_gram = style_grams[layer]
            _, d, h, w = target_features[layer].shape
            style_loss += style_weights[layer] * torch.mean((target_gram - style_gram) ** 2) / (d * h * w)

        total_loss = content_weight * content_loss + style_weight * style_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if i % print_after == 0 or i == epochs:
            img = imcnvt(target)
            output_path = os.path.join(output_dir, f"styled_{i}.png")
            plt.imsave(output_path, img)
            print(f"[INFO] Epoch {i}/{epochs} - Loss: {total_loss:.4f}")
    return output_path
