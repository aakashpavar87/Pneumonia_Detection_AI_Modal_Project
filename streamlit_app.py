import streamlit as st
from PIL import Image
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, InputLayer
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os,json
from keras.metrics import AUC

# Setting random seeds for reproducibility
import random
import tensorflow as tf

im = Image.open("favicon.ico")
st.set_page_config(
    page_title="Pneumonia Detection with Python",
    page_icon=im,
    layout="wide",
)
st.logo('static/lungs.png')

# Setting random seeds for reproducibility
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
hyper_epochs = 50
hyper_channels = 1  # Grayscale
model_path = "cnn_model.h5"  # File to save/load the trained model

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


# Save the trained model to a file
def save_trained_model(model, history):
    model.save(model_path)
    history_path = model_path.replace(".h5", "_history.json")

    with open(history_path, 'w') as f:
        json.dump(history.history, f)
    st.success(f"Model saved to {model_path}")


# Load the trained model from the file
def load_trained_model():
    if os.path.exists(model_path):
        model = load_model(model_path)
        st.success("Model loaded successfully.")
        
        history_path = model_path.replace(".h5", "_history.json")
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                history = json.load(f)
            st.success("Training history loaded successfully.")
            return model, history
        else:
            st.warning("Training history file not found.")
            return model, None
    else:
        st.error(f"Model file not found at {model_path}. Please train the model first.")
        return None, None


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
    save_trained_model(cnn, history) # Save model after training
    return cnn, history


# Predict on uploaded X-Ray image
def predict_pneumonia(uploaded_file, model):
    from keras.preprocessing import image
    img = image.load_img(uploaded_file, target_size=(hyper_dimension, hyper_dimension), color_mode='grayscale')
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    st.image(img_array[0].squeeze(), caption="Preprocessed X-Ray", use_container_width=True, clamp=True)
    prediction = model.predict(img_array)
    return prediction[0][0]  # Probability of having pneumonia


# Create charts and metrics for analysis
def create_charts(cnn, cnn_history):
    train_loss = cnn_history['loss']
    val_loss = cnn_history['val_loss']
    train_auc = cnn_history['auc']  # AUC for training
    val_auc = cnn_history['val_auc']  # AUC for validation

    # Confusion matrix predictions
    y_true = test_generator.classes
    y_pred_prob = cnn.predict(test_generator, steps=len(test_generator)).T[0]
    y_pred = (y_pred_prob > 0.5).astype(int)

    # Chart 1: Train vs Validation Loss
    st.write("### Training vs. Validation Loss")
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    st.pyplot(plt)

    # Chart 2: Train vs Validation AUC
    st.write("### Training vs. Validation AUC")
    plt.figure(figsize=(10, 6))
    plt.plot(train_auc, label='Training AUC')
    plt.plot(val_auc, label='Validation AUC')
    plt.xlabel("Epochs")
    plt.ylabel("AUC")
    plt.legend()
    st.pyplot(plt)

    # Chart 3: Confusion Matrix
    st.write("### Confusion Matrix")
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Pneumonia'], yticklabels=['Normal', 'Pneumonia'])
    st.pyplot(plt)

    # Chart 4: ROC Curve
    st.write("### ROC Curve")
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc_score(y_true, y_pred_prob):.2f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    st.pyplot(plt)

    st.write("### Summary Statistics 🔢")
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


# Streamlit application layout
def main():
    image_url = "./app/static/lungs.png"
    title = "Chest X-Ray Pneumonia Detection and Dataset Analysis"
    st.markdown(
        f"""
            <h1 style="display: flex; align-items: center; gap: 10px;">
                {title}
                <img src="{image_url}" alt="lung image" style="height: 50px;">
            </h1>
            """,
        unsafe_allow_html=True,
    )
    choice = st.sidebar.radio("Select an Option", ["Train Model", "Check X-Ray for Pneumonia", "Analyze Dataset"])

    if choice == "Train Model":
        st.header("Train Model")
        if st.button("Start Training"):
            st.write("Training the model...")
            cnn_model, cnn_history = train_cnn_model()
            st.write("Training completed!")
            # create_charts(cnn_model, cnn_history)  # Show charts after training

    elif choice == "Check X-Ray for Pneumonia":
        st.header("Check X-Ray for Pneumonia")
        uploaded_file = st.file_uploader("Upload a Chest X-Ray Image (PNG/JPEG)", type=["png", "jpg", "jpeg"])
        model,history = load_trained_model()

        if uploaded_file is not None and model:
            st.image(uploaded_file, caption="Uploaded X-Ray", use_container_width=True)
            st.write("Analyzing the uploaded X-Ray...")
            result = predict_pneumonia(uploaded_file, model)
            if result > 0.5:
                st.write("### Result: Pneumonia Detected 😔")
                st.write(f"Confidence: {result:.2%}")
            else:
                st.write("### Result: No Pneumonia Detected 😊")
                st.write(f"Confidence: {100 - result * 100:.2%}")

    elif choice == "Analyze Dataset":
        st.header("Analyze Dataset")
        model, history = load_trained_model()
        if model and history:
            create_charts(model, history)
        else:
            st.warning("Please train the model first.")


# Run the app
if __name__ == "__main__":
    main()
