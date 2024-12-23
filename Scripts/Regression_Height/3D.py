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
import pandas as pd
from io import BytesIO
from urllib.parse import quote_plus
import numpy as np

np.random.seed(1)
import random

random.seed(1)
import json
import pandas as pd
from sklearn.decomposition import PCA
from pyproj import Proj, transform
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tensorflow import keras
from tensorflow.keras import layers, models
import tensorflow as tf
import plotly.io as pio
import plotly.graph_objects as go
from datetime import datetime
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from numpy.polynomial.polynomial import Polynomial
from scipy.spatial import cKDTree
from tqdm import tqdm

# Set environment variable for deterministic operations
os.environ['TF_DETERMINISTIC_OPS'] = '1'

# Set global seeds
tf.random.set_seed(1)
np.random.seed(1)
random.seed(1)
# Set the seed for random operations to ensure deterministic behavior
tf.random.set_seed(1)

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


# Check for data leakage
def check_overlap(set1, set2, name1="Set1", name2="Set2"):
    overlap = set(map(tuple, set1.reshape(set1.shape[0], -1))) & set(map(tuple, set2.reshape(set2.shape[0], -1)))
    if overlap:
        raise ValueError(f"Data leakage detected between {name1} and {name2}: {len(overlap)} overlapping samples!")
    else:
        print(f"No overlap detected between {name1} and {name2}.")


# %% Defs
def idw_interpolation(x, y, z, xv, yv, power=2):
    tree = cKDTree(np.c_[x, y])
    distances, indices = tree.query(np.c_[xv.ravel(), yv.ravel()], k=4)
    weights = 1 / distances ** power
    weights /= weights.sum(axis=1, keepdims=True)
    z_interpolated = np.sum(weights * z[indices], axis=1)
    return z_interpolated.reshape(xv.shape)


def projection3d_with_interpolation(
        lidar_raw,
        interpolate=True,
        vertical_fill=True,
        pool_size=None
):
    """
    Process 3D LiDAR data into a voxel grid with options for interpolation, vertical fill, and max pooling.

    Args:
        lidar_raw (ndarray): Input LiDAR data with shape (n, 3) where columns are x, y, z.
        interpolate (bool): Whether to interpolate z-values in the xy-plane.
        vertical_fill (bool): Whether to fill all voxels below an occupied voxel.
        pool_size (int or None): If provided, perform max pooling with the given pool size.

    Returns:
        ndarray: A boolean 3D grid representing the processed LiDAR data.
    """
    # Separate the data into x, y, z
    x, y, z = lidar_raw[:, 0], lidar_raw[:, 1], lidar_raw[:, 2]

    # Scale coordinates to centimeters
    x = x * 100
    y = y * 100
    z = z * 100

    # Set minimum of each axis to 0 cm
    x -= np.min(x)
    y -= np.min(y)
    z -= np.min(z)

    # Perform PCA for dominant direction in the xy-plane
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

    # Set resolution for grid (in cm per pixel for x and y axes)
    x_resolution = (np.max(rotated_x) - np.min(rotated_x)) / 300
    y_resolution = (np.max(rotated_y) - np.min(rotated_y)) / 100

    # Initialize the 3D boolean grid
    grid_3d = np.zeros((100, 300, 150), dtype=bool)  # 100 x 300 x 150 grid

    # Create grid for interpolation
    x_min, y_min = np.min(rotated_x), np.min(rotated_y)
    x_grid = np.linspace(x_min, np.max(rotated_x), 300)
    y_grid = np.linspace(y_min, np.max(rotated_y), 100)
    xv, yv = np.meshgrid(x_grid, y_grid)

    if interpolate:
        z_interpolated = idw_interpolation(rotated_x, rotated_y, z, xv, yv)

        # Fill the grid using interpolated z-values
        for grid_y in range(100):
            for grid_x in range(300):
                z_max = np.nanmax(z_interpolated[grid_y, grid_x])
                if not np.isnan(z_max):
                    grid_z = int(z_max)
                    if 0 <= grid_z < 150:
                        grid_3d[grid_y, grid_x, grid_z] = True

    # Preserve maximum z-values
    for i in range(len(rotated_x)):
        grid_x = int((rotated_x[i] - x_min) / x_resolution)
        grid_y = int((rotated_y[i] - y_min) / y_resolution)
        grid_z = int(z[i])  # Use z directly as an index

        if 0 <= grid_x < 300 and 0 <= grid_y < 100 and 0 <= grid_z < 150:
            grid_3d[grid_y, grid_x, grid_z] = True

    if vertical_fill:
        # Vertical fill: Set all voxels below an occupied voxel to True
        for x in range(grid_3d.shape[1]):  # Iterate over x-axis
            for y in range(grid_3d.shape[0]):  # Iterate over y-axis
                for z in range(1, grid_3d.shape[2]):  # Start from second layer
                    if grid_3d[y, x, z]:  # If current voxel is True
                        grid_3d[y, x, :z] = True  # Set all below to True

    if pool_size:
        # Perform max pooling
        pooled_shape = (
            grid_3d.shape[0] // pool_size,
            grid_3d.shape[1] // pool_size,
            grid_3d.shape[2] // pool_size
        )
        pooled_grid = np.zeros(pooled_shape, dtype=bool)

        for i in range(pooled_shape[0]):
            for j in range(pooled_shape[1]):
                for k in range(pooled_shape[2]):
                    pooled_grid[i, j, k] = np.max(
                        grid_3d[
                        i * pool_size: (i + 1) * pool_size,
                        j * pool_size: (j + 1) * pool_size,
                        k * pool_size: (k + 1) * pool_size
                        ]
                    )
        return pooled_grid

    return grid_3d


# %% Access the database with read_only access
with open('/gpfs/fs7/aafc/phenocart/PhenomicsProjects/UFPSGPSCProject/5_Data/MongoDB/config.json') as config_file:
    config = json.load(config_file)

# %% Connect with MongoDB server
username = quote_plus(config['mongodb_username'])
password = quote_plus(config['mongodb_password'])

# Construct the connection string using the escaped username and password
uri = f"mongodb://{username}:{password}@localhost:27018/"

# Replace "localhost" with the hostname or IP address of your MongoDB server
client = pymongo.MongoClient(uri)
db = client["UFPS"]
fs = GridFS(db)
collection = db["Data"]

# Trigger server selection to check if connection is successful
print("Attempting to ping the MongoDB server...")
db.command('ping')  # Sending a ping command to the database
print("Ping to MongoDB server successful.")

# %% Get documents
variable = 'height'
criteria = {}
documents = collection.find(criteria)
maxpool = 2

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
        resized_grid = projection3d_with_interpolation(lidar_raw, pool_size=maxpool)
        x_train.append(resized_grid)
        y_train.append(document[variable])
    except Exception as e:
        exceptions.append((document['_id'], str(e)))  # Log the document ID and exception message

# Step 3: Print exceptions after processing
if exceptions:
    print("\nExceptions encountered during processing:")
    for doc_id, error_message in exceptions:
        print(f"Document ID {doc_id}: {error_message}")

x_train_data = np.reshape(np.asarray(x_train), (len(x_train), 50, 150, 75, 1))
y_train_data = np.reshape(np.asarray(y_train), (len(y_train), 1))


# %% Path to the model file
# Model Preparation Function
def prepare_model_without_output(input_shape):
    inputs = layers.Input(shape=input_shape)

    def residual_block(x, filters, kernel_size, strides=(1, 1, 1)):
        shortcut = x
        x = layers.Conv3D(filters, kernel_size=kernel_size, strides=strides, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv3D(filters, kernel_size=(1, 1, 1), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)

        # Adjust shortcut dimensions if necessary
        if shortcut.shape[1:] != x.shape[1:]:
            shortcut = layers.Conv3D(filters, kernel_size=(1, 1, 1), strides=strides, padding='same')(shortcut)
        x = layers.Add()([shortcut, x])
        x = layers.ReLU()(x)
        return x

    # Data augmentation layer
    def random_flip_3d(inputs):
        def augment(inputs):
            inputs = tf.cond(
                tf.random.uniform(()) > 0.5, lambda: tf.reverse(inputs, axis=[1]), lambda: inputs)
            inputs = tf.cond(
                tf.random.uniform(()) > 0.5, lambda: tf.reverse(inputs, axis=[2]), lambda: inputs)
            inputs = tf.cond(
                tf.random.uniform(()) > 0.5, lambda: tf.reverse(inputs, axis=[3]), lambda: inputs)
            return inputs

        return tf.keras.layers.Lambda(augment)(inputs)

    x = random_flip_3d(inputs)
    x = layers.Conv3D(32, (3, 3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = residual_block(x, 64, (3, 3, 3))
    x = layers.MaxPooling3D(pool_size=(2, 2, 2))(x)
    x = residual_block(x, 128, (3, 3, 3))
    x = layers.MaxPooling3D(pool_size=(2, 2, 2))(x)
    x = residual_block(x, 256, (3, 3, 3))
    x = layers.MaxPooling3D(pool_size=(2, 2, 2))(x)
    x = residual_block(x, 512, (3, 3, 3))
    x = layers.MaxPooling3D(pool_size=(2, 2, 2))(x)
    x = layers.GlobalAveragePooling3D()(x)

    # Fully connected layers
    x = layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)

    return models.Model(inputs=inputs, outputs=x)


# Initialize the model without the output layer
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    base_model = prepare_model_without_output(input_shape=(50, 150, 75, 1))  # Shared base model

# DataFrames to store results across all repetitions and iterations
results = []

# Initialize variables to track the best metrics and model
best_r2 = -float('inf')  # Start with the lowest possible R²
best_rmse = float('inf')  # Start with the highest possible RMSE
best_model_path = None

for rep in range(1, 51):  # Run repetitions
    print(f"Starting Repetition {rep}...")

    # Perform a single split for train, validation, and test sets
    train_indices, test_indices = train_test_split(
        np.arange(len(x_train_data)), test_size=0.2, shuffle=True
    )
    train_indices, val_indices = train_test_split(
        train_indices, test_size=0.2, shuffle=True
    )

    # Subset the data
    X_train, X_val, X_test = x_train_data[train_indices], x_train_data[val_indices], x_train_data[test_indices]
    Y_train, Y_val, Y_test = y_train_data[train_indices], y_train_data[val_indices], y_train_data[test_indices]

    # Check for data leakage
    check_overlap(X_train, X_test, "Training", "Test")
    check_overlap(X_train, X_val, "Training", "Validation")
    check_overlap(X_val, X_test, "Validation", "Test")

    with strategy.scope():
        x = base_model.output
        new_output = layers.Dense(1, activation='linear')(x)
        model = models.Model(inputs=base_model.input, outputs=new_output)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                      loss='mean_squared_error',
                      metrics=['mean_absolute_error'])
        # Set up early stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)

        # Train the model
        history = model.fit(
            X_train,
            Y_train,
            validation_data=(X_val, Y_val),
            epochs=1000,
            batch_size=64,
            callbacks=[early_stopping],
            verbose=1
        )

        # R2 and RMSE calculation
        y_pred = model.predict(X_test).flatten()
        y_true = Y_test.flatten()

        # Calculate R2
        R2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)

        # Calculate RMSE
        RMSE = np.sqrt(np.mean((y_true - y_pred) ** 2))

        # Calculate MAPE (Mean Absolute Percentage Error)
        MAPE = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

        # Update best model tracking
        if R2 > best_r2 or RMSE < best_rmse:
            best_r2 = R2
            best_rmse = RMSE
            best_model_path = '/gpfs/fs7/aafc/phenocart/PhenomicsProjects/UFPSGPSCProject/6_Output/5_Final Results/2024/Best_3D_HeightEstimation_Model.h5'
            model.save(best_model_path)  # Save the best model
            print(f"New best model saved with R²: {R2:.2f}, RMSE: {RMSE:.2f}")

        # Append R2, RMSE, and file paths to the results
        results.append({
            'Repetition': rep,
            'R2 Score (%)': (R2 * 100),
            'RMSE': RMSE,
            'MAPE (%)': MAPE
        })

        # Print only the last result in the list
        last_result = results[-1]
        print("Repetition:", last_result['Repetition'])
        print("R2 Score (%):", round(last_result['R2 Score (%)'], 2))
        print("RMSE:", round(last_result['RMSE'], 2))
        print("MAPE (%):", round(last_result['MAPE (%)'], 2))

        # Save results to an Excel file at the end
        results_df = pd.DataFrame(results)
        results_file_path = save_file_with_timestamp(
            '/gpfs/fs7/aafc/phenocart/PhenomicsProjects/UFPSGPSCProject/6_Output/5_Final Results/2024/',
            '3D_HeightEstimation_Results',
            'xlsx'
        )
        results_df.to_excel(results_file_path, index=False)

# At the end of all repetitions, display the best model path
print(f"The best model was saved to {best_model_path}")
