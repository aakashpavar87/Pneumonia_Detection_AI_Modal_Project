# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score
from keras import backend as K
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, InputLayer, Activation
from keras.regularizers import l2
from keras.metrics import AUC
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set a seed value for reproducibility
seed_value = 42

# 1. Set `PYTHONHASHSEED` environment variable to a fixed value
import os
os.environ['PYTHONHASHSEED'] = str(seed_value)

# 2. Set Python's built-in pseudo-random generator to a fixed value
import random
random.seed(seed_value)

# 3. Set NumPy's pseudo-random generator to a fixed value
np.random.seed(seed_value)

# 4. Set TensorFlow's pseudo-random generator to a fixed value
tf.random.set_seed(seed_value)

# 5. Remove the session-related code (TensorFlow 2.x manages sessions automatically)
# For TensorFlow 2.x, you don't need to manually configure a session. Just set seeds for reproducibility.

## Set file paths to image files
project_path = "C:\\Users\\Aakash Pavar\\Desktop\\Fourth_Year_AI_Project"
train_path = project_path + "\\chest_xray\\train\\"
val_path = project_path + "\\chest_xray\\val\\"
test_path = project_path + "\\chest_xray\\test\\"

## Set up hyperparameters that will be used later
hyper_dimension = 64
hyper_batch_size = 128
hyper_epochs = 100
hyper_channels = 1  # This corresponds to 'grayscale' mode
hyper_mode = 'grayscale'  # To read images in grayscale

## Generate batches of image data (train, validation, and test) with data augmentation
train_datagen = ImageDataGenerator(rescale=1.0/255.0, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1.0/255.0)
test_datagen = ImageDataGenerator(rescale=1.0/255.0)

# Flow images from the directories with data augmentation
train_generator = train_datagen.flow_from_directory(
    directory=train_path,
    target_size=(hyper_dimension, hyper_dimension),
    batch_size=hyper_batch_size,
    color_mode=hyper_mode,
    class_mode='binary',  # Since it's a binary classification problem
    seed=seed_value
)

val_generator = val_datagen.flow_from_directory(
    directory=val_path,
    target_size=(hyper_dimension, hyper_dimension),
    batch_size=hyper_batch_size,
    class_mode='binary',
    color_mode=hyper_mode,
    shuffle=False,  # For validation, no need to shuffle
    seed=seed_value
)

test_generator = test_datagen.flow_from_directory(
    directory=test_path,
    target_size=(hyper_dimension, hyper_dimension),
    batch_size=hyper_batch_size,
    class_mode='binary',
    color_mode=hyper_mode,
    shuffle=False,  # For testing, no need to shuffle
    seed=seed_value
)

# Reset the test generator to start from the first batch
test_generator.reset()

# Building the CNN model
cnn = Sequential()
cnn.add(InputLayer(input_shape=(hyper_dimension, hyper_dimension, hyper_channels)))

# Adding convolutional and pooling layers
cnn.add(Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(MaxPooling2D(pool_size=(2, 2)))

cnn.add(Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(MaxPooling2D(pool_size=(2, 2)))

cnn.add(Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(MaxPooling2D(pool_size=(2, 2)))

# Flattening the output from convolutional layers and adding dense layers
cnn.add(Flatten())
cnn.add(Dense(activation='relu', units=128))
cnn.add(Dense(activation='sigmoid', units=1))  # Binary classification

# Compile the model with optimizer, loss function, and metrics
cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=[AUC()])

# Fit the model using the training data and validate with validation data
cnn_model = cnn.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=10,  # Adjust number of epochs if necessary
    validation_data=val_generator,
    validation_steps=len(val_generator),
    verbose=2  # Verbose=2 provides more detailed output during training
)

# Function to create performance charts for the model
def create_charts(cnn, cnn_model):
    # Extract training and validation loss
    train_loss = cnn_model.history['loss']
    val_loss = cnn_model.history['val_loss']

    # Extract training and validation AUC scores
    train_auc_name = list(cnn_model.history.keys())[3]  # Get train AUC key
    val_auc_name = list(cnn_model.history.keys())[1]  # Get validation AUC key
    train_auc = cnn_model.history[train_auc_name]
    val_auc = cnn_model.history[val_auc_name]

    # Predict on test data
    y_true = test_generator.classes  # True labels
    Y_pred = cnn.predict(test_generator, steps=len(test_generator))  # Predicted probabilities
    y_pred = (Y_pred > 0.5).T[0]  # Convert probabilities to binary predictions
    y_pred_prob = Y_pred.T[0]

    # Plotting
    fig = plt.figure(figsize=(12, 10))

    # Plot 1: Train vs. Validation Loss
    plt.subplot(2, 2, 1)
    plt.title("Training vs. Validation Loss")
    plt.plot(train_loss, label='training loss')
    plt.plot(val_loss, label='validation loss')
    plt.xlabel("Number of Epochs", size=14)
    plt.legend()

    # Plot 2: Train vs. Validation AUC
    plt.subplot(2, 2, 2)
    plt.title("Training vs. Validation AUC Score")
    plt.plot(train_auc, label='training auc')
    plt.plot(val_auc, label='validation auc')
    plt.xlabel("Number of Epochs", size=14)
    plt.legend()

    # Plot 3: Confusion Matrix
    plt.subplot(2, 2, 3)
    # Set up confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    names = ['True Negatives', 'False Positives', 'False Negatives', 'True Positives']
    counts = ['{0:0.0f}'.format(value) for value in cm.flatten()]
    percentages = ['{0:.2%}'.format(value) for value in cm.flatten() / np.sum(cm)]
    labels = [f'{v1}\n{v2}' for v1, v2 in zip(names, percentages)]
    labels = np.asarray(labels).reshape(2, 2)
    ticklabels = ['Normal', 'Pneumonia']

    # Create confusion matrix heatmap
    sns.set(font_scale=1.4)
    ax = sns.heatmap(cm, annot=labels, fmt='', cmap='Oranges', xticklabels=ticklabels, yticklabels=ticklabels)
    plt.xticks(size=12)
    plt.yticks(size=12)
    plt.title("Confusion Matrix", size=16)
    plt.xlabel("Predicted", size=14)
    plt.ylabel("Actual", size=14)

    # Plot 4: ROC Curve
    plt.subplot(2, 2, 4)
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
    auc_score = roc_auc_score(y_true, y_pred_prob)
    plt.title('ROC Curve', size=16)
    plt.plot([0, 1], [0, 1], 'k--', label="Random (AUC = 50%)")
    plt.plot(fpr, tpr, label='CNN (AUC = {:.2f}%)'.format(auc_score * 100))
    plt.xlabel('False Positive Rate', size=14)
    plt.ylabel('True Positive Rate', size=14)
    plt.legend(loc='best')

    # Finalize and show plots
    plt.tight_layout()

    plt.show()  # Add this line to display the plots

    # Calculate summary statistics
    TN, FP, FN, TP = cm.ravel()  # True negatives, false positives, etc.
    accuracy = (TP + TN) / np.sum(cm)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    specificity = TN / (TN + FP)
    f1 = 2 * precision * recall / (precision + recall)
    stats_summary = '[Summary Statistics]\nAccuracy = {:.2%} | Precision = {:.2%} | Recall = {:.2%} | Specificity = {:.2%} | F1 Score = {:.2%}'.format(
        accuracy, precision, recall, specificity, f1)
    print(stats_summary)

# Call the function to create performance charts and print statistics
create_charts(cnn, cnn_model)
