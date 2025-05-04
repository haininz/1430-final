import torch
import os
from matplotlib import pyplot as plt
from model import get_model 
from custom_dataset import get_transform
from main import load_latest_model, predict_single_image

from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay


def collect_predictions(image_folder, label):

    """Calculates and records the predicted and true labels for each image in the folder"""

    print(f"working on the folder for label: {label}")
    # List to hold the predictions and true labels
    true_labels = []
    predictions = []

    # Setting up parameters for my model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(device)
    model = load_latest_model(model, device, './models')
    transform = get_transform()

    # Collecting the image paths from the folders
    image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

    # Running the model for each image and recording the predicted label along with the true label
    for path in image_paths:
        try:
            predicted_label, probabilities = predict_single_image(path, model, device, transform)
            print(predicted_label)
            predictions.append(predicted_label)
            true_labels.append(label)

        except Exception  as e:
            print(f"Error processing {path}: {e}")

    return true_labels, predictions



def  evaluate_model(ai_folder, real_folder):

    """Puts everything together to present performance results"""

    # Get the true and predicted labels for each folder
    true_ai, pred_ai = collect_predictions(ai_folder, label = 1)
    true_real, pred_real = collect_predictions(real_folder, label = 0)

    # Join the lists of true and predicted labels
    y_true = true_ai + true_real

    # The map is used since my model returns results as real or fake and not an int 0 or 1
    label_map = {"real": 0, "fake": 1}
    y_pred = [label_map[label] for label in pred_ai] + [label_map[label] for label in pred_real]

    # Show results via confusion matrix and classification report
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels = ["Real", "AI"])
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.show()

    print("Classificatiion Report")
    print(classification_report(y_true, y_pred, target_names=["Real", "AI"]))


# Run on the folders from the Kaggle Dataset
evaluate_model(ai_folder="AiArtData", real_folder="RealArt")




