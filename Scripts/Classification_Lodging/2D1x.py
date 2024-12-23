import sys

sys.path.append('/gpfs/fs7/aafc/phenocart/PhenomicsProjects/cleanqc/')

from pathlib import Path
from natsort import natsorted
import os
import cleanqc
import kmeans1d
from instruments.locatedInstrument import located_instrument
import pymongo
from gridfs import GridFS
from datetime import datetime
import pandas as pd
from io import BytesIO
from urllib.parse import quote_plus
import numpy as np
import json
from sklearn.decomposition import PCA
from pyproj import Proj, transform
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
import tensorflow as tf

tf.random.set_seed(1)
from imblearn.over_sampling import SMOTE
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.losses import categorical_crossentropy
from sklearn.utils.class_weight import compute_class_weight
import plotly.graph_objects as go
import plotly.io as pio
from numpy.polynomial.polynomial import Polynomial
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, f1_score
from tqdm import tqdm

# Set Plotly renderer to SVG
pio.renderers.default = 'svg'

# Initialize lists to store R2 values and filenames
results = []


# Define function for timestamp
def get_timestamp():
    return datetime.utcnow().strftime('%Y%m%d_%H%M%S')


# Function to save files with UTC timestamp in their names
def save_file_with_timestamp(base_path, filename, extension):
    timestamp = get_timestamp()
    return f"{base_path}{filename}_{timestamp}.{extension}"


def projection1x(lidar_raw):
    # Separate the data into x, y, z
    x, y, z = lidar_raw[:, 0], lidar_raw[:, 1], lidar_raw[:, 2]

    # Set minimum of each axis to 0 cm
    x -= np.min(x)
    y -= np.min(y)
    z -= np.min(z)

    # Scale coordinates to centimeters
    x = x * 100
    y = y * 100
    z = z * 100

    # PCA for dominant direction in X-Y plane (optional for rotation)
    points = np.vstack((x, y)).T
    pca = PCA(n_components=2)
    pca.fit(points)
    angle = np.arctan2(pca.components_[0, 1], pca.components_[0, 0])

    # Rotate points to align with grid axes
    rotation_matrix = np.array([
        [np.cos(-angle), -np.sin(-angle)],
        [np.sin(-angle), np.cos(-angle)]
    ])
    rotated_points = points @ rotation_matrix.T
    rotated_x, rotated_y = rotated_points[:, 0], rotated_points[:, 1]

    # Define resolution for the X-Y plane
    xy_resolution = 1  # Adjusted to 1 cm for X-Y plane

    # Define boundaries for the X-Y plane
    x_min, x_max = np.min(rotated_x), np.max(rotated_x)
    y_min, y_max = np.min(rotated_y), np.max(rotated_y)

    # Calculate grid size
    grid_x_size_xy = int((x_max - x_min) / xy_resolution) + 1
    grid_y_size_xy = int((y_max - y_min) / xy_resolution) + 1

    # Initialize grid with NaNs
    grid_xy = np.full((grid_y_size_xy, grid_x_size_xy), np.nan, dtype=np.float32)

    # Project Z onto X-Y plane using maximum aggregation
    for i in range(len(rotated_x)):
        grid_x = int((rotated_x[i] - x_min) / xy_resolution)
        grid_y = int((rotated_y[i] - y_min) / xy_resolution)
        if np.isnan(grid_xy[grid_y, grid_x]):
            grid_xy[grid_y, grid_x] = z[i]
        else:
            grid_xy[grid_y, grid_x] = max(grid_xy[grid_y, grid_x], z[i])

    # Replace NaNs with zero
    grid_xy = np.nan_to_num(grid_xy, nan=0.0)

    # Optionally resize grid for display purposes
    resized_grid_xy = cv2.resize(grid_xy, (300, 100), interpolation=cv2.INTER_NEAREST)

    return resized_grid_xy


# Iterative Class Mapping
def map_classes(y_train, iteration):
    if iteration == 1:  # Binary classification: 0 -> 0, 1-9 -> 1
        return np.where(y_train == 0, 0, 1)
    elif iteration == 2:  # Three-class classification: 0 -> 0, 1-4 -> 1, 5-9 -> 2
        return np.select(
            [y_train == 0, (y_train >= 1) & (y_train <= 4), (y_train >= 5) & (y_train <= 9)],
            [0, 1, 2]
        )
    elif iteration == 3:  # Three-class classification: 0 -> 0, 1-4 -> 1, 5-9 -> 2
        return np.select(
            [y_train == 0, (y_train >= 1) & (y_train <= 2), (y_train >= 3) & (y_train <= 4),
             (y_train >= 5) & (y_train <= 6), (y_train >= 7) & (y_train <= 9)],
            [0, 1, 2, 3, 4]
        )
    elif iteration == 4:  # Full classification: 0-9 as separate classes
        return np.select(
            [y_train == 0, (y_train == 1), (y_train == 2), (y_train == 3), (y_train == 4), (y_train == 5),
             (y_train == 6), (y_train == 7),
             (y_train >= 8) & (y_train <= 9)],
            [0, 1, 2, 3, 4, 5, 6, 7, 8]
        )
    else:
        raise ValueError("Invalid iteration number")


def quadratic_weighted_kappa(y_true, y_pred, num_classes):
    """
    Calculate Quadratic Weighted Kappa (QWK).

    Args:
    y_true (array-like): True labels.
    y_pred (array-like): Predicted labels.
    num_classes (int): Total number of classes.

    Returns:
    float: QWK score.

    """
    # Confusion matrix (Observed agreement)
    O = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))

    # Marginal probabilities for true and predicted
    row_marginals = np.sum(O, axis=1)
    col_marginals = np.sum(O, axis=0)

    # Expected agreement
    total = np.sum(O)
    E = np.outer(row_marginals, col_marginals) / total

    # Quadratic weights
    W = np.zeros((num_classes, num_classes))
    for i in range(num_classes):
        for j in range(num_classes):
            W[i, j] = ((i - j) ** 2) / ((num_classes - 1) ** 2)

    # QWK formula
    O_weighted = np.sum(W * O)
    E_weighted = np.sum(W * E)

    kappa = 1 - (O_weighted / E_weighted)
    return kappa


# %% Database Access
with open('/gpfs/fs7/aafc/phenocart/PhenomicsProjects/UFPSGPSCProject/5_Data/MongoDB/config.json') as config_file:
    config = json.load(config_file)

username = quote_plus(config['mongodb_username'])
password = quote_plus(config['mongodb_password'])
uri = f"mongodb://{username}:{password}@localhost:27018/"
client = pymongo.MongoClient(uri)
db = client["UFPS"]
fs = GridFS(db)
collection = db["Data"]

# Trigger server selection to check if connection is successful
print("Attempting to ping the MongoDB server...")
db.command('ping')  # Sending a ping command to the database
print("Ping to MongoDB server successful.")

# %% Get documents
variable = 'lodging'
criteria = {}
documents = collection.find(criteria)

# Step 1: Filter documents that satisfy the condition
valid_documents = [
    document for document in tqdm(documents, desc="Filtering valid documents")
    if
    'lidar_id' in document and variable in document and not pd.isna(document[variable]) and document[variable] not in (
    None, '', [], {})
]
print(f"\nNumber of valid documents: {len(valid_documents)}")

# Step 2: Process the valid documents
x_train, y_train = [], []
exceptions = []  # Collect exceptions here

for document in tqdm(valid_documents, desc="Processing valid documents"):
    try:
        gridout = fs.get(document['lidar_id'])
        lidar_raw = np.load(BytesIO(gridout.read()))
        resized_grid = projection1x(lidar_raw)
        x_train.append(resized_grid)
        y_train.append(document[variable])
    except Exception as e:
        exceptions.append((document['_id'], str(e)))  # Log the document ID and exception message

# Step 3: Print exceptions after processing
if exceptions:
    print("\nExceptions encountered during processing:")
    for doc_id, error_message in exceptions:
        print(f"Document ID {doc_id}: {error_message}")

# Assuming x_train_data has a 4D shape: (samples, timesteps, features, channels)
x_train_data = np.reshape(np.asarray(x_train), (len(x_train), 300, 100, 1))
y_train_data = np.reshape(np.asarray(y_train), (len(y_train), 1))


# %% Path to the model file
# Model Preparation Function
def prepare_model_without_output(input_shape):
    inputs = layers.Input(shape=input_shape)

    # Squeeze-and-Excitation block
    def squeeze_excite_block(input, ratio=16):
        filters = input.shape[-1]
        se = layers.GlobalAveragePooling2D()(input)
        se = layers.Dense(filters // ratio, activation='relu')(se)
        se = layers.Dense(filters, activation='sigmoid')(se)
        se = layers.Reshape([1, 1, filters])(se)
        return layers.Multiply()([input, se])

    # Residual Depthwise Separable Block with SE block and Leaky ReLU
    def residual_depthwise_separable_block(x, filters, kernel_size, strides):
        shortcut = x
        x = layers.DepthwiseConv2D(kernel_size=kernel_size, strides=strides, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=0.1)(x)
        x = layers.Conv2D(filters, kernel_size=(1, 1), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = squeeze_excite_block(x)  # Add SE block
        if shortcut.shape == x.shape:
            x = layers.Add()([shortcut, x])
        else:
            shortcut = layers.Conv2D(filters, kernel_size=(1, 1), strides=strides, padding='same')(shortcut)
            x = layers.Add()([shortcut, x])
        x = layers.LeakyReLU(alpha=0.1)(x)
        return x

    # Stacking Residual Blocks with Attention
    x = residual_depthwise_separable_block(inputs, 16, (3, 3), (1, 1))
    x = residual_depthwise_separable_block(x, 32, (3, 3), (2, 2))
    x = residual_depthwise_separable_block(x, 64, (3, 3), (1, 1))
    x = residual_depthwise_separable_block(x, 128, (3, 3), (2, 2))
    x = residual_depthwise_separable_block(x, 256, (3, 3), (2, 2))
    x = residual_depthwise_separable_block(x, 512, (3, 3), (1, 1))
    x = residual_depthwise_separable_block(x, 1024, (3, 3), (2, 2))
    x = residual_depthwise_separable_block(x, 2048, (3, 3), (2, 2))

    # Attention Layer
    x = layers.Attention()([x, x])

    # Global Pooling and Fully Connected Layers with Dropout
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = layers.Dense(16, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)

    return models.Model(inputs=inputs, outputs=x)


# Initialize the model without the output layer
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    base_model = prepare_model_without_output(input_shape=(300, 100, 1))  # Shared base model

# DataFrames to store results across all repetitions and iterations
all_accuracies = pd.DataFrame()
all_qwks = pd.DataFrame()
all_macro_f1s = pd.DataFrame()  # New DataFrame for Macro-F1 scores

# Initialize dictionaries to store the best metrics for each iteration
best_metrics = {iteration: {'accuracy': -np.inf, 'qwk': -np.inf, 'macro_f1': -np.inf} for iteration in range(1, 5)}

# Path for saving models
model_save_path = '/gpfs/fs7/aafc/phenocart/PhenomicsProjects/UFPSGPSCProject/6_Output/Best_Models/'

# Create the directory if it doesn't exist
os.makedirs(model_save_path, exist_ok=True)

for rep in range(1, 26):  # Run 10 repetitions
    print(f"Starting Repetition {rep}...")

    rep_accuracies = []  # Store accuracies for this repetition
    rep_qwks = []  # Store QWK scores for this repetition
    rep_macro_f1s = []  # Store Macro-F1 scores for this repetition

    for iteration in range(1, 5):  # Iterations for progressive fine-tuning
        print(f"  Starting Iteration {iteration}...")

        # Assuming `x_train_data` and `y_train_data` are your input features and target labels

        # Step 1: Map classes for the current iteration
        y_train_mapped = map_classes(y_train_data, iteration)
        num_classes = len(np.unique(y_train_mapped))
        print(f"  Number of classes for iteration {iteration}: {num_classes}")

        # Step 2: Perform a single split for train, validation, and test sets
        train_indices, test_indices = train_test_split(
            np.arange(len(x_train_data)), test_size=0.2, shuffle=True
        )
        train_indices, val_indices = train_test_split(
            train_indices, test_size=0.2, shuffle=True
        )

        # Subset the data
        X_train_raw, X_val, X_test = x_train_data[train_indices], x_train_data[val_indices], x_train_data[test_indices]
        Y_train_raw, Y_val, Y_test = y_train_mapped[train_indices], y_train_mapped[val_indices], y_train_mapped[
            test_indices]

        # Step 3: Apply SMOTE to the training set
        # Flatten the training data
        X_train_flattened = X_train_raw.reshape(len(X_train_raw), -1)

        # Use SMOTE to balance the training set
        smote = SMOTE(k_neighbors=1)  # Use k_neighbors=1 for rare classes
        X_train_resampled, Y_train_resampled = smote.fit_resample(X_train_flattened, Y_train_raw)

        # Reshape back to original dimensions
        X_train_resampled = X_train_resampled.reshape(-1, X_train_raw.shape[1], X_train_raw.shape[2], 1)

        # Step 4: Convert the resampled labels to one-hot encoding
        Y_train_res_onehot = to_categorical(Y_train_resampled, num_classes=num_classes)
        Y_val_onehot = to_categorical(Y_val, num_classes=num_classes)
        Y_test_onehot = to_categorical(Y_test, num_classes=num_classes)

        # Step 5: Final split for training and validation
        X_train_final, X_val_final, Y_train_final, Y_val_final = train_test_split(
            X_train_resampled, Y_train_res_onehot, test_size=0.2
        )

        # Add the output layer for the current number of classes
        with strategy.scope():
            x = base_model.output
            new_output = layers.Dense(num_classes, activation='softmax')(x)
            model = models.Model(inputs=base_model.input, outputs=new_output)
            # model.summary()
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])

        # Data Augmentation
        datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True)
        train_generator = datagen.flow(X_train_final, Y_train_final, batch_size=64)

        # Train the model
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True)
        lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=25, verbose=1,
                                                            min_lr=1e-6)
        history = model.fit(
            X_train_final, Y_train_final,
            validation_data=(X_val, Y_val_onehot),
            epochs=1000,
            batch_size=64,
            callbacks=[early_stopping, lr_scheduler],
            verbose=1
        )

        # Evaluate the model
        y_pred = np.argmax(model.predict(X_test), axis=1)
        y_true = Y_test

        # Accuracy, QWK, and Macro-F1
        accuracy = accuracy_score(y_true, y_pred)
        qwk = quadratic_weighted_kappa(y_true, y_pred, num_classes)
        macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)  # Calculate Macro-F1 Score

        # Confusion matrix
        conf_matrix = confusion_matrix(y_true, y_pred)

        # Append scores to lists
        rep_accuracies.append(accuracy * 100)
        rep_qwks.append(qwk)
        rep_macro_f1s.append(macro_f1 * 100)

        print(
            f"  Iteration {iteration} Complete. Accuracy: {accuracy * 100:.2f}%, QWK: {qwk:.4f}, Macro-F1: {macro_f1 * 100:.2f}%")

        # Save results to Excel
        accuracy_data = pd.DataFrame({
            'Iteration': [iteration],
            'Accuracy (%)': [accuracy * 100],
            'QWK': [qwk],
            'Macro-F1 (%)': [macro_f1 * 100]
        })

        # Check if current metrics are better for this iteration
        current_metrics = best_metrics[iteration]
        if (
                accuracy > current_metrics['accuracy']
                or qwk > current_metrics['qwk']
                or macro_f1 > current_metrics['macro_f1']
        ):
            # Update the best metrics for this iteration
            best_metrics[iteration]['accuracy'] = max(current_metrics['accuracy'], accuracy)
            best_metrics[iteration]['qwk'] = max(current_metrics['qwk'], qwk)
            best_metrics[iteration]['macro_f1'] = max(current_metrics['macro_f1'], macro_f1)

            # Save the model (overwrite the previous model for this iteration)
            model_filename = f"{model_save_path}2D1x_best_model_for_iteration_{iteration}.h5"
            model.save(model_filename)
            print(f"  Model saved as {model_filename} due to improvement in metrics.")

        # Add repetition results to the DataFrame
    all_accuracies[f'Repetition {rep}'] = rep_accuracies
    all_qwks[f'Repetition {rep}'] = rep_qwks
    all_macro_f1s[f'Repetition {rep}'] = rep_macro_f1s  # Add Macro-F1 scores

    # Save all accuracies, QWK scores, and Macro-F1 scores to Excel
    output_file = '{model_save_path}2D1x_all_results.xlsx'

    # Dynamically set the index length based on iterations
    all_accuracies.index = [f'Iteration {i}' for i in range(1, len(rep_accuracies) + 1)]
    all_qwks.index = [f'Iteration {i}' for i in range(1, len(rep_qwks) + 1)]
    all_macro_f1s.index = [f'Iteration {i}' for i in range(1, len(rep_macro_f1s) + 1)]

    with pd.ExcelWriter(output_file) as writer:
        all_accuracies.to_excel(writer, sheet_name='Accuracies')
        all_qwks.to_excel(writer, sheet_name='QWK Scores')
        all_macro_f1s.to_excel(writer, sheet_name='Macro-F1 Scores')  # Save Macro-F1 scores

    print(f"All results saved to {output_file}")
