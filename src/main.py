import os
import sys
import rawpy
import numpy as np
import torch
import torchvision
from torchvision import transforms
from PIL import Image


def load_model():
    model = torchvision.models.resnet18(pretrained=True)
    model.eval()
    return model


def process_raw_image(raw_path):
    with rawpy.imread(raw_path) as raw:
        rgb_image = raw.postprocess()
        return rgb_image


def classify_images(model, transform, image_folder):
    images = {}  # {element: [list of image paths]}
    for image_name in os.listdir(image_folder):
        if image_name.lower().endswith(".raw"):
            image_path = os.path.join(image_folder, image_name)
            rgb_image = process_raw_image(image_path)
            pil_image = Image.fromarray(rgb_image)

            input_tensor = transform(pil_image).unsqueeze(0)
            output = model(input_tensor)
            _, preds = torch.max(output, 1)
            element = preds.item()

            if element in images:
                images[element].append(image_path)
            else:
                images[element] = [image_path]

            if len(images[element]) == 2:
                os.makedirs(os.path.join("output", str(element)), exist_ok=True)
                for img_path in images[element]:
                    os.rename(img_path, os.path.join("output", str(element), os.path.basename(img_path)))
    return images


def main():
    image_folder = "input_images"

    model = load_model()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    images = classify_images(model, transform, image_folder)

    for element, image_paths in images.items():
        print(f"{element}: {', '.join(image_paths)}")


if __name__ == "__main__":
    main()