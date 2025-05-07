from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import cv2
from huggingface_hub import from_pretrained_keras
import os

ai_imgs = os.listdir("AiArtData")
real_imgs = os.listdir("RealArt")
k_model = from_pretrained_keras('C:\\Users\\nadia\\mobilenet_v2_fake_image_detection')
all_labels = list(np.zeros(len(real_imgs))) + list(np.ones(len(ai_imgs)))
all_predicted = []

for img_str in ai_imgs:
    image = cv2.imread("AiArtData/" + img_str)
    image = cv2.resize(image, (128, 128))
    image = image.astype('float32') / 255
    image = np.expand_dims(image, axis=0)
    preds = k_model.predict(image)
    result = preds[0][0]
    if result > 0.5:
        all_predicted.append(1)
        with open("ai_correctly_classified.txt", "a") as f:
            f.write(img_str + "\n")
    else:
        all_predicted.append(0)
        with open("ai_misclassified_as_natural.txt", "a") as f:
            f.write(img_str+ "\n")

for img_str in real_imgs:
    image = cv2.imread("RealArt/" + img_str)
    image = cv2.resize(image, (128, 128))
    image = image.astype('float32') / 255
    image = np.expand_dims(image, axis=0)
    preds = k_model.predict(image)
    result = preds[0][0]
    if result > 0.5:
        all_predicted.append(1)
        with open("natural_misclassified_as_ai.txt", "a") as f:
            f.write(img_str+ "\n")
    else:
        all_predicted.append(0)
        with open("natural_correctly_classified.txt", "a") as f:
            f.write(img_str+ "\n")

# Convert all_labels and all_predicted to numpy arrays if they are not already

all_labels = np.array(all_labels)
all_predicted = np.array(all_predicted)

accuracy = accuracy_score(all_labels, all_predicted)
precision = precision_score(all_labels, all_predicted, average='macro')
recall = recall_score(all_labels, all_predicted, average='macro')
f1 = f1_score(all_labels, all_predicted, average='macro')

print(f'Accuracy: {accuracy * 100:.2f}%')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')
