import pymongo
from gridfs import GridFS
from datetime import datetime
import pandas as pd
from io import BytesIO
from urllib.parse import quote_plus
import numpy as np
np.random.seed(1)
import random
random.seed(1)
import json
from sklearn.decomposition import PCA
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
import tensorflow as tf
tf.random.set_seed(1)

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import plotly.io as pio
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


# Check for data leakage
def check_overlap(set1, set2, name1="Set1", name2="Set2"):
    overlap = set(map(tuple, set1.reshape(set1.shape[0], -1))) & set(map(tuple, set2.reshape(set2.shape[0], -1)))
    if overlap:
        raise ValueError(f"Data leakage detected between {name1} and {name2}: {len(overlap)} overlapping samples!")
    else:
        print(f"No overlap detected between {name1} and {name2}.")


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
variable = 'height'
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
    x = residual_depthwise_separable_block(x, 256, (3, 3), (1, 1))
    x = residual_depthwise_separable_block(x, 512, (3, 3), (2, 2))
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

        # Data Augmentation
        datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True)
        train_generator = datagen.flow(X_train, Y_train, batch_size=16)

        # Train the model
        history = model.fit(
            train_generator,
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
            best_model_path = '/gpfs/fs7/aafc/phenocart/PhenomicsProjects/UFPSGPSCProject/6_Output/5_Final Results/2024/Best_2D1x_HeightEstimation_Model.h5'
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
            '2D1x_HeightEstimation_Results',
            'xlsx'
        )
        results_df.to_excel(results_file_path, index=False)

# At the end of all repetitions, display the best model path
print(f"The best model was saved to {best_model_path}")
