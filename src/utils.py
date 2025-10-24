import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, accuracy_score
from tensorflow.keras.datasets import fashion_mnist
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from tensorflow.keras import models, layers

# Create output folder
os.makedirs("output", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Helper: Save model results
def save_results(model_name, y_true, y_pred):
    # Confusion Matrix
    disp = ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
    plt.title(f"{model_name} - Confusion Matrix")
    plt.savefig(f"output/confusion_matrix_{model_name}.png")
    plt.close()

    # Classification Report
    report = classification_report(y_true, y_pred)
    with open(f"output/classification_report_{model_name}.txt", "w") as f:
        f.write(report)

    print(f"✅ {model_name} results saved to /output/")

# Load dataset & normalize
def load_data():
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x_train = x_train.reshape(len(x_train), -1) / 255.0
    x_test = x_test.reshape(len(x_test), -1) / 255.0
    return x_train, x_test, y_train, y_test

# SVM Model
def train_svm(x_train, x_test, y_train, y_test):
    model = SVC()
    model.fit(x_train[:10000], y_train[:10000])
    
    y_pred = model.predict(x_test[:5000])
    accuracy = accuracy_score(y_test[:5000], y_pred)
    print("SVM Accuracy:", accuracy)

    save_results("svm", y_test[:5000], y_pred)

# Random Forest Model
def train_random_forest(x_train, x_test, y_train, y_test):
    model = RandomForestClassifier()
    model.fit(x_train[:10000], y_train[:10000])
    
    y_pred = model.predict(x_test[:5000])
    accuracy = accuracy_score(y_test[:5000], y_pred)
    print("RF Accuracy:", accuracy)

    save_results("random_forest", y_test[:5000], y_pred)

# PCA + SVM
def train_pca_svm(x_train, x_test, y_train, y_test):
    pca = PCA(n_components=150)
    x_train_pca = pca.fit_transform(x_train[:10000])
    x_test_pca = pca.transform(x_test[:5000])
    
    model = SVC()
    model.fit(x_train_pca, y_train[:10000])

    y_pred = model.predict(x_test_pca)
    accuracy = accuracy_score(y_test[:5000], y_pred)
    print("PCA+SVM Accuracy:", accuracy)

    save_results("pca_svm", y_test[:5000], y_pred)

# Deep Learning Model
def train_deep_learning_model(x_train, x_test, y_train, y_test):
    x_train = x_train.reshape(-1, 28, 28)
    x_test = x_test.reshape(-1, 28, 28)

    model = models.Sequential([
        layers.Flatten(input_shape=(28, 28)),
        layers.Dense(256, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=5, batch_size=128)

    y_pred_probs = model.predict(x_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    accuracy = accuracy_score(y_test, y_pred)
    print("Deep Learning Accuracy:", accuracy)

    model.save("models/fashion_model.h5")
    print("Model Saved to /models/fashion_model.h5")

    save_results("deep_learning", y_test, y_pred)
