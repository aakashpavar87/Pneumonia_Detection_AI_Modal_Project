import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, InputLayer
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from keras.metrics import AUC

# Setting random seeds for reproducibility
import random
import tensorflow as tf

seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

# File paths for dataset
project_path = "C:\\Users\\Aakash Pavar\\Desktop\\Fourth_Year_AI_Project"
train_path = project_path + "\\chest_xray\\train\\"
val_path = project_path + "\\chest_xray\\val\\"
test_path = project_path + "\\chest_xray\\test\\"

# Hyperparameters
hyper_dimension = 64
hyper_batch_size = 128
hyper_epochs = 10
hyper_channels = 1  # Grayscale

# Data augmentation and generators
train_datagen = ImageDataGenerator(rescale=1.0 / 255.0, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

train_generator = train_datagen.flow_from_directory(
    directory=train_path,
    target_size=(hyper_dimension, hyper_dimension),
    batch_size=hyper_batch_size,
    color_mode='grayscale',
    class_mode='binary',
    seed=seed_value
)

val_generator = val_datagen.flow_from_directory(
    directory=val_path,
    target_size=(hyper_dimension, hyper_dimension),
    batch_size=hyper_batch_size,
    color_mode='grayscale',
    class_mode='binary',
    shuffle=False,
    seed=seed_value
)

test_generator = test_datagen.flow_from_directory(
    directory=test_path,
    target_size=(hyper_dimension, hyper_dimension),
    batch_size=hyper_batch_size,
    color_mode='grayscale',
    class_mode='binary',
    shuffle=False,
    seed=seed_value
)


# Build the CNN model
def build_cnn_model():
    cnn = Sequential()
    cnn.add(InputLayer(input_shape=(hyper_dimension, hyper_dimension, hyper_channels)))
    cnn.add(Conv2D(filters=32, kernel_size=3, activation='relu'))
    cnn.add(MaxPooling2D(pool_size=(2, 2)))
    cnn.add(Conv2D(filters=32, kernel_size=3, activation='relu'))
    cnn.add(MaxPooling2D(pool_size=(2, 2)))
    cnn.add(Conv2D(filters=32, kernel_size=3, activation='relu'))
    cnn.add(MaxPooling2D(pool_size=(2, 2)))
    cnn.add(Flatten())
    cnn.add(Dense(activation='relu', units=128))
    cnn.add(Dense(activation='sigmoid', units=1))  # Binary classification
    cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=[AUC()])
    return cnn


# Train the CNN model
@st.cache_resource
def train_cnn_model():
    cnn = build_cnn_model()
    history = cnn.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=hyper_epochs,
        validation_data=val_generator,
        validation_steps=len(val_generator),
        verbose=1
    )
    return cnn, history


cnn_model, cnn_history = train_cnn_model()


# Predict on uploaded X-Ray image
def predict_pneumonia(uploaded_file):
    # Load the image
    from keras.preprocessing import image
    img = image.load_img(uploaded_file, target_size=(hyper_dimension, hyper_dimension), color_mode='grayscale')
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    # Predict using the CNN model
    prediction = cnn_model.predict(img_array)
    return prediction[0][0]  # Probability of having pneumonia


def create_charts(cnn, cnn_history):
    # Extract training and validation loss
    train_loss = cnn_history.history['loss']
    val_loss = cnn_history.history['val_loss']

    # Extract training and validation AUC scores
    train_auc_name = list(cnn_history.history.keys())[3]  # Get train AUC key
    val_auc_name = list(cnn_history.history.keys())[1]  # Get validation AUC key
    train_auc = cnn_history.history[train_auc_name]
    val_auc = cnn_history.history[val_auc_name]

    # Predict on test data
    y_true = test_generator.classes
    Y_pred = cnn.predict(test_generator, steps=len(test_generator))
    y_pred = (Y_pred > 0.5).T[0]
    y_pred_prob = Y_pred.T[0]

    # **Chart 1: Train vs. Validation Loss**
    st.write("### Training vs. Validation Loss")
    fig1, ax1 = plt.subplots()
    ax1.plot(train_loss, label='Training Loss')
    ax1.plot(val_loss, label='Validation Loss')
    ax1.set_title("Loss per Epoch")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend()
    st.pyplot(fig1)

    # **Chart 2: Train vs. Validation AUC**
    st.write("### Training vs. Validation AUC Score")
    fig2, ax2 = plt.subplots()
    ax2.plot(train_auc, label='Training AUC')
    ax2.plot(val_auc, label='Validation AUC')
    ax2.set_title("AUC per Epoch")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("AUC")
    ax2.legend()
    st.pyplot(fig2)

    # **Chart 3: Confusion Matrix**
    st.write("### Confusion Matrix")
    cm = confusion_matrix(y_true, y_pred)
    ticklabels = ['Normal', 'Pneumonia']
    fig3, ax3 = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=ticklabels, yticklabels=ticklabels, ax=ax3)
    ax3.set_title("Confusion Matrix")
    ax3.set_xlabel("Predicted")
    ax3.set_ylabel("Actual")
    st.pyplot(fig3)

    # **Chart 4: ROC Curve**
    st.write("### ROC Curve")
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
    auc_score = roc_auc_score(y_true, y_pred_prob)
    fig4, ax4 = plt.subplots()
    ax4.plot(fpr, tpr, label=f"CNN (AUC = {auc_score:.2f})")
    ax4.plot([0, 1], [0, 1], 'k--', label="Random Guess (AUC = 0.50)")
    ax4.set_title("ROC Curve")
    ax4.set_xlabel("False Positive Rate")
    ax4.set_ylabel("True Positive Rate")
    ax4.legend()
    st.pyplot(fig4)

    # **Statistics**
    st.write("### Summary Statistics")
    TN, FP, FN, TP = cm.ravel()
    accuracy = (TP + TN) / np.sum(cm)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    specificity = TN / (TN + FP)
    f1 = 2 * (precision * recall) / (precision + recall)

    st.write(f"- **Accuracy**: {accuracy:.2%}")
    st.write(f"- **Precision**: {precision:.2%}")
    st.write(f"- **Recall**: {recall:.2%}")
    st.write(f"- **Specificity**: {specificity:.2%}")
    st.write(f"- **F1 Score**: {f1:.2%}")


# Analyze dataset
def analyze_dataset():
    st.title("X-Ray Dataset Analysis")
    st.write("Analyzing dataset using the trained model.")

    # Display paths
    st.write(f"Training Path: {train_path}")
    st.write(f"Validation Path: {val_path}")
    st.write(f"Test Path: {test_path}")

    # Example: Show a sample chart
    create_charts(cnn_model, cnn_history)


# Streamlit application layout
def main():
    st.title("Chest X-Ray Pneumonia Detection and Dataset Analysis")
    choice = st.sidebar.radio("Select an Option", ["Check X-Ray for Pneumonia", "Analyze Dataset"])

    if choice == "Check X-Ray for Pneumonia":
        st.header("Check X-Ray for Pneumonia")
        uploaded_file = st.file_uploader("Upload a Chest X-Ray Image (PNG/JPEG)", type=["png", "jpg", "jpeg"])

        if uploaded_file is not None:
            st.image(uploaded_file, caption="Uploaded X-Ray", use_container_width=True)
            st.write("Analyzing the uploaded X-Ray...")
            result = predict_pneumonia(uploaded_file)
            if result > 0.5:
                st.write("### Result: Pneumonia Detected")
                st.write(f"Confidence: {result:.2%}")
            else:
                st.write("### Result: No Pneumonia Detected")
                st.write(f"Confidence: {100 - result * 100:.2%}")

    elif choice == "Analyze Dataset":
        analyze_dataset()


# Run the app
if __name__ == "__main__":
    main()
