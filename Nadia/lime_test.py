import os
import cv2
import matplotlib.pyplot as plt
import skimage
import numpy as np
from datetime import datetime
from huggingface_hub import from_pretrained_keras
from lime import lime_image
import keras

# thank you cs1430 staff for the easily borrowed LIME code!

def LIME_explainer(model, path, timestamp):
    """
    This function takes in a trained model and a path to an image and outputs 4
    visual explanations using the LIME model
    """

    save_directory = "lime_explainer_images" + os.sep + timestamp
    if not os.path.exists("lime_explainer_images"):
        os.mkdir("lime_explainer_images")
    if not os.path.exists(save_directory):
        os.mkdir(save_directory)
    image_index = 0

    def image_and_mask(title, positive_only=True, num_features=5,
                       hide_rest=True):
        nonlocal image_index

        temp, mask = explanation.get_image_and_mask(
            explanation.top_labels[0], positive_only=positive_only,
            num_features=num_features, hide_rest=hide_rest)
        plt.imshow(skimage.segmentation.mark_boundaries(temp / 2 + 0.5, mask))
        plt.title(title)

        image_save_path = save_directory + os.sep + str(image_index) + ".png"
        plt.savefig(image_save_path, dpi=300, bbox_inches='tight')
        plt.show()

        image_index += 1

    # Read the image and preprocess it as before
    image = cv2.imread(path)
    if len(image.shape) == 2:
        image = np.stack([image, image, image], axis=-1)
    image = cv2.resize(image, (128, 128))
    image = image.astype('float32') / 255

    explainer = lime_image.LimeImageExplainer()

    explanation = explainer.explain_instance(
        image.astype('double'), model.predict, top_labels=5, hide_color=0,
        num_samples=500)

    # The top 5 superpixels that are most positive towards the class with the
    # rest of the image hidden
    image_and_mask("Top 5 superpixels", positive_only=True, num_features=5,
                   hide_rest=True)

    # The top 5 superpixels with the rest of the image present
    image_and_mask("Top 5 with the rest of the image present",
                   positive_only=True, num_features=5, hide_rest=False)

    # The 'pros and cons' (pros in green, cons in red)
    image_and_mask("Pros(green) and Cons(red)",
                   positive_only=False, num_features=10, hide_rest=False)

    # Select the same class explained on the figures above.
    ind = explanation.top_labels[0]
    # Map each explanation weight to the corresponding superpixel
    dict_heatmap = dict(explanation.local_exp[ind])
    heatmap = np.vectorize(dict_heatmap.get)(explanation.segments)
    plt.imshow(heatmap, cmap='RdBu', vmin=-heatmap.max(), vmax=heatmap.max())
    plt.colorbar()
    plt.title("Map each explanation weight to the corresponding superpixel")

    image_save_path = save_directory + os.sep + str(image_index) + ".png"
    plt.savefig(image_save_path, dpi=300, bbox_inches='tight')
    plt.show()


model = from_pretrained_keras('C:\\Users\\nadia\\mobilenet_v2_fake_image_detection')
time_now = datetime.now()
timestamp = time_now.strftime("%m%d%y-%H%M%S")
LIME_explainer(model, "laputa_real.jpg", timestamp)