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
    image_paths = []

    # Setting up parameters for my model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(device)
    model = load_latest_model(model, device, './models')
    transform = get_transform()

    # Collecting the image paths from the folders
    image_paths_all = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

    # Running the model for each image and recording the predicted label along with the true label
    for path in image_paths_all:
        try:
            predicted_label, probabilities = predict_single_image(path, model, device, transform)
            print(predicted_label)
            predictions.append(predicted_label)
            true_labels.append(label)
            image_paths.append(path)

        except Exception  as e:
            print(f"Error processing {path}: {e}")

    return true_labels, predictions, image_paths



def  evaluate_model(ai_folder, real_folder):

    """Puts everything together to present performance results"""

    # Get the true and predicted labels for each folder
    true_ai, pred_ai, paths_ai = collect_predictions(ai_folder, label = 1)
    true_real, pred_real, path_real = collect_predictions(real_folder, label = 0)

    # Join the lists of true and predicted labels
    y_true = true_ai + true_real

    # The map is used since my model returns results as real or fake and not an int 0 or 1
    label_map = {"real": 0, "fake": 1}
    y_pred = [label_map[label] for label in pred_ai] + [label_map[label] for label in pred_real]

    all_paths = paths_ai + path_real

    ai_correct, ai_incorrect = [], []

    real_correct, real_incorrect = [], []

    for true, pred, path in  zip(y_true, y_pred, all_paths):
        if true == 1:
            if pred == true:
                ai_correct.append(path)
            else:
                ai_incorrect.append(path)
        else:
            if pred == true:
                real_correct.append(path)
            else:
                real_incorrect.append(path)

    
    with open("ai_correctly_classified.txt", "w") as f:
        f.writelines([os.path.basename(p) + "\n" for p in ai_correct])

    with open("ai_misclassified_as_natural.txt", "w") as f:
        f.writelines([os.path.basename(p)+ "\n" for p in ai_incorrect])

    with open("natural_correctly_classified.txt", "w") as f:
        f.writelines([os.path.basename(p)+ "\n" for p in real_correct])

    with open("natural_misclassified_as_ai.txt", "w") as f:
        f.writelines([os.path.basename(p) + "\n" for p in real_incorrect])


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




