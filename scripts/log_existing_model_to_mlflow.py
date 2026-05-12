import mlflow
import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import os

# --- CONFIG ---
MODEL_PATH = "models/vgg16_politician_final.h5"  # Change to your model
MODEL_NAME = "VGG16"
CLASSES = ['aitzaz_ahsan', 'asif_ali_zardari', 'asif_ghafoor', 'bilawal_bhutto_zardari', 'fawad_chaudhry', 'fazlur_rehman', 'hamza_shehbaz', 'hina_rabbani_khar', 'imran_khan', 'khawaja_asif', 'marzam_nawaz', 'nawaz_sharif', 'pervez_elahi', 'shah_muhammad_qureshi', 'shehbaz_sharif', 'siraj_ul_haq']
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
TEST_DIR = "dataset/test"  # Update if needed

# --- LOAD MODEL ---
model = tf.keras.models.load_model(MODEL_PATH)

def load_test_data():
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    test_gen = datagen.flow_from_directory(
        TEST_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    return test_gen

def plot_confusion_matrix(cm, classes, fname):
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(fname)
    plt.close()

if __name__ == "__main__":
    mlflow.set_experiment("Politician_Classifier")
    with mlflow.start_run(run_name=MODEL_NAME):
        test_gen = load_test_data()
        y_true = test_gen.classes
        y_pred = model.predict(test_gen)
        y_pred_classes = np.argmax(y_pred, axis=1)

        # Only use present classes for metrics
        present_labels = sorted(list(set(y_true)))
        present_class_names = [CLASSES[i] for i in present_labels]

        # Metrics
        report = classification_report(
            y_true, y_pred_classes,
            labels=present_labels,
            target_names=present_class_names,
            output_dict=True
        )
        acc = report['accuracy']
        mlflow.log_metric("accuracy", acc)
        for cls in present_class_names:
            mlflow.log_metric(f"precision_{cls}", report[cls]['precision'])
            mlflow.log_metric(f"recall_{cls}", report[cls]['recall'])
            mlflow.log_metric(f"f1_{cls}", report[cls]['f1-score'])

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred_classes, labels=present_labels)
        cm_path = f"confusion_matrix_{MODEL_NAME}.png"
        plot_confusion_matrix(cm, present_class_names, cm_path)
        mlflow.log_artifact(cm_path)
        os.remove(cm_path)

        # Top 5 misclassified samples
        misclassified_idxs = np.where(y_true != y_pred_classes)[0]
        top5 = misclassified_idxs[:5]
        with open("top5_misclassified.txt", "w") as f:
            for idx in top5:
                fname = test_gen.filenames[idx]
                true_cls = CLASSES[y_true[idx]]
                pred_cls = CLASSES[y_pred_classes[idx]]
                f.write(f"{fname}: true={true_cls}, pred={pred_cls}\n")
        mlflow.log_artifact("top5_misclassified.txt")
        os.remove("top5_misclassified.txt")

        print("MLflow logging complete for", MODEL_NAME)
