import torch

from skimage.io import imread
from skimage.transform import resize
from lime import lime_image
from skimage.segmentation import mark_boundaries, slic
from matplotlib import pyplot as plt
import numpy as np

from PIL import Image
from model import get_model 
from custom_dataset import get_transform
from main import load_latest_model


def predict_lime_v(images, model, device, transform):

    # print("received", len(images), "Images")

    model.eval()
    tensors = []

    for img in images: 

        image = Image.fromarray(img) if isinstance(img, np.ndarray) else img

        transformed_image = transform(image).unsqueeze(0).to(device)

        tensors.append(transformed_image)

    batch = torch.cat(tensors, dim=0)

    # print("Batch size", batch.shape)

    with torch.no_grad():
        outputs = model(batch)
        # Get all predictions
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=1)


    return probabilities.numpy()
    

image_path = "man_ai.png"
image_np = imread(image_path)

if image_np.dtype != np.uint8:
    print(f"Image in format {image_np.dtype}")
    image_np = (image_np * 255).astype(np.uint8)


explainer = lime_image.LimeImageExplainer()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = get_model(device)
model = load_latest_model(model, device, './models')


explanation = explainer.explain_instance(
    image_np,
    classifier_fn=lambda images: predict_lime_v(images, model, device, get_transform()),
    top_labels = 1,
    hide_color = 0,
    num_samples = 500,
    segmentation_fn = lambda image: slic(image, n_segments=50, compactness=10, sigma=1),
    random_seed=0
)


temp, mask = explanation.get_image_and_mask(
    label = explanation.top_labels[0],
    positive_only = False,
    hide_rest = False,
    num_features=5
)

plt.imshow(mark_boundaries(temp, mask))
plt.axis("off")
plt.title("LIME EXPLANATION")
plt.show()

ind = explanation.top_labels[0]
# Map each explanation weight to the corresponding superpixel
dict_heatmap = dict(explanation.local_exp[ind])
heatmap = np.vectorize(dict_heatmap.get)(explanation.segments)
plt.imshow(heatmap, cmap='RdBu', vmin=-heatmap.max(), vmax=heatmap.max())
plt.colorbar()
plt.title("Map each explanation weight to the corresponding superpixel")
plt.show()
