import PIL
import numpy as np
import tensorflow as tf
from skimage.transform import resize

from huggingface_hub import from_pretrained_keras

import matplotlib.pyplot as plt

model = from_pretrained_keras('C:\\Users\\nadia\\mobilenet_v2_fake_image_detection')
model.summary()

img = PIL.Image.open("burning_ghibli_fake.jpeg") # Load image

img = np.asarray(img) # Convert the image to a Numpy array

img = resize(img, (128, 128, 3), anti_aliasing=True) # Resize the image and normalize the values (to be between 0.0 and 1.0)
plt.imshow(img, cmap='gray', vmin=0.0, vmax=1.0) # Display image (make sure we're looking at the right thing)
plt.show()

# The Keras model expects images in a 4D array with dimensions (sample, height, width, channel)

images = np.array([img]) # Keras expects more than one image (in Numpy array), so convert image(s) to such array
print(images.shape) # Print dimensions of inference input

preds = model.predict(images) # Inference

model.layers[-1].activation = None # For either algorithm, we need to remove the Softmax activation function of the last layer

# Based on: https://github.com/keisen/tf-keras-vis/blob/master/tf_keras_vis/saliency.py
def get_saliency_map(img_array, model):

  img_tensor = tf.convert_to_tensor(img_array) # Gradient calculation requires input to be a tensor

  # Do a forward pass of model with image and track the computations on the "tape"
  with tf.GradientTape(watch_accessed_variables=False, persistent=True) as tape:

    tape.watch(img_tensor) # Compute (non-softmax) outputs of model with given image
    outputs = model(img_tensor, training=False)

    score = outputs[:, 0] # Get score (predicted value) of actual class

  grads = tape.gradient(score, img_tensor) # Compute gradients of the loss with respect to the input image

  grads_disp = [np.max(g, axis=-1) for g in grads] # Finds max value in each color channel of the gradient (should be grayscale for this demo)

  grad_disp = grads_disp[0] # There should be only one gradient heatmap for this demo

  grad_disp = tf.abs(grad_disp) # The absolute value of the gradient shows the effect of change at each pixel. Source: https://christophm.github.io/interpretable-ml-book/pixel-attribution.html

  heatmap_min = np.min(grad_disp) # Normalize to between 0 and 1 (use epsilon, a very small float, to prevent divide-by-zero error)
  heatmap_max = np.max(grad_disp)
  heatmap = (grad_disp - heatmap_min) / (heatmap_max - heatmap_min + tf.keras.backend.epsilon())

  return heatmap.numpy()


saliency_map = get_saliency_map(images, model) # Generate saliency map for the given input image


idx = 0 # Overlay the saliency map on top of the original input image
ax = plt.subplot()
ax.imshow(img, vmin=0.0, vmax=1.0)
ax.imshow(saliency_map, cmap='magma', alpha=0.25)
plt.show()