import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from transformers import pipeline, AutoImageProcessor


# Wrap the model so that only logits are returned (needed for GradCAM compatibility)
class CustomModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super(CustomModelWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        outputs = self.model(x)
        return outputs.logits

# Reshape Swin Transformer outputs (batch_size, num_patches, hidden_dim)
# into (batch_size, hidden_dim, height, width) so GradCAM can process it
def reshape_transform(tensor, height=7, width=7):
    result = tensor.permute(0, 2, 1)
    result = result.reshape(result.size(0), result.size(1), height, width)
    return result

pipe = pipeline("image-classification", model="umm-maybe/AI-image-detector")
base_model = pipe.model

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
base_model = base_model.to(device)
base_model.eval()

model = CustomModelWrapper(base_model)

# Preprocessing pipeline
processor = AutoImageProcessor.from_pretrained("umm-maybe/AI-image-detector")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=processor.image_mean, std=processor.image_std)
])

# Load and preprocess image
img_path = "img3.jpg"
img = Image.open(img_path).convert("RGB")
img = img.resize((224, 224))
input_tensor = transform(img).unsqueeze(0).to(device)

# Inference to get predicted class
with torch.no_grad():
    outputs = model(input_tensor)
predicted_class = outputs.argmax(dim=1).item()

# Define target layer(s) from the model where GradCAM will extract feature maps and gradients
# Changing target_layers significantly changes the resulting heatmap focus
# target_layers = [model.model.swin.encoder.layers[-1].blocks[-1].attention.output.dense]
target_layers = [model.model.swin.encoder.layers[0].blocks[-1].attention.output.dense]

cam = GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform)

targets = [ClassifierOutputTarget(predicted_class)]

grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
grayscale_cam = grayscale_cam[0, :]

rgb_img = np.array(img) / 255.0
visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

plt.imshow(visualization)
plt.axis('off')

prediction_label = "Artificial" if predicted_class == 0 else "Natural"
confidence = torch.nn.functional.softmax(outputs, dim=1)[0][predicted_class].item()
plt.title(f"Prediction: {prediction_label} ({confidence*100:.1f}%)", fontsize=14, color='white', backgroundcolor='black')

plt.show()
