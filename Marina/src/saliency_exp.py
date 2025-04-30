import torch

from skimage.io import imread
from skimage.transform import resize
from lime import lime_image
from skimage.segmentation import mark_boundaries, slic
from matplotlib import pyplot as plt
import numpy as np
from captum.attr import Saliency

from PIL import Image
from model import get_model 
from custom_dataset import get_transform
from main import load_latest_model

image_path = "man_ai.png"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = get_model(device)

model = load_latest_model(model, device, './models')

model.eval()

class LogitsWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x).logits 

wrapped_model = LogitsWrapper(model)

transform = get_transform()

image = Image.open(image_path).convert("RGB")

transformed_image = transform(image).unsqueeze(0).to(device) 

transformed_image = transformed_image.requires_grad_()

# Make a prediction

with torch.no_grad():
    outputs = model(transformed_image)
    logits =  outputs.logits
    _, predicted = logits.max(1)
    predicted_class = predicted.item()
   

saliency = Saliency(wrapped_model)

attributions = saliency.attribute(transformed_image, target=predicted_class)

saliency_map= attributions.squeeze().cpu().detach().numpy()
saliency_map= np.abs(saliency_map).sum(axis=0)  # Sum across RGB channels
saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())  # Normalize to [0, 1]

original_image = np.asarray(image.resize((224, 224))) / 255.0

# Plot
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.title("Original")
plt.imshow(original_image)
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Saliency Map")
plt.imshow(original_image)
plt.imshow(saliency_map, cmap="hot", alpha=0.5)  # Overlay heatmap
plt.axis("off")

plt.tight_layout()
plt.show()


summary = model.summary()
print(summary)


