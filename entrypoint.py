import sys
import os
import torch
from PIL import Image
from torchvision import transforms
from model.generator import UNetGenerator

def enhance_image(model, image_path, output_path):
    image = Image.open(image_path).convert("L")
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)[0]

    output_image = transforms.ToPILImage()(output)
    output_image.save(output_path)

def main():
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    os.makedirs(output_dir, exist_ok=True)

    model = UNetGenerator()
    model.load_state_dict(torch.load("results/model_best.pth", map_location="cpu"))
    model.eval()

    for file in os.listdir(input_dir):
        if file.endswith(".png"):
            input_path = os.path.join(input_dir, file)
            output_path = os.path.join(output_dir, file)
            enhance_image(model, input_path, output_path)

if __name__ == "__main__":
    main()
