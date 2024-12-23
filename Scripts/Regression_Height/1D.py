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
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
import tensorflow as tf
tf.random.set_seed(1)
from tqdm import tqdm
import plotly.io as pio

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


def height_distribution(lidar_raw, height_bins=150):
    """
    Create a 1D height distribution array for LiDAR height data (z-values).

    Args:
        lidar_raw (np.ndarray): Nx3 array with LiDAR data (x, y, z).
        height_bins (int): Number of bins in the height dimension (default is 150).

    Returns:
        np.ndarray: 1D array representing the distribution of points across height bins.
    """
    # Extract z values (heights) from LiDAR data
    z = lidar_raw[:, 2]
    z = (z - np.min(z)) * 100

    # Normalize z values to fit within 1 to 150 cm range
    z_min, z_max = 0, 150  # Set the fixed height range
    z_scaled = np.clip(z, z_min, z_max)  # Clip values outside the range
    z_indices = ((z_scaled - z_min) / (z_max - z_min) * (height_bins - 1)).astype(int)  # Scale to bin indices

    # Initialize a 1D grid for height distribution
    height_distribution = np.zeros(height_bins, dtype=int)

    # Count the number of points in each height bin
    for idx in z_indices:
        height_distribution[idx] += 1

    return height_distribution


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
        resized_grid = height_distribution(lidar_raw)
        x_train.append(resized_grid)
        y_train.append(document[variable])
    except Exception as e:
        exceptions.append((document['_id'], str(e)))  # Log the document ID and exception message

# Step 3: Print exceptions after processing
if exceptions:
    print("\nExceptions encountered during processing:")
    for doc_id, error_message in exceptions:
        print(f"Document ID {doc_id}: {error_message}")

x_train_data = np.reshape(np.asarray(x_train), (len(x_train), 150, 1))
y_train_data = np.reshape(np.asarray(y_train), (len(y_train), 1))

# %% Path to the model file
# Model Preparation Function
def prepare_model_without_output(input_shape):
    inputs = layers.Input(shape=input_shape)

    def residual_depthwise_separable_block(x, filters, kernel_size, strides):
        shortcut = x
        x = layers.DepthwiseConv1D(kernel_size=kernel_size, strides=strides, padding='same')(x)
        x = layers.LayerNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv1D(filters, kernel_size=1, padding='same', activation='relu')(x)
        x = layers.LayerNormalization()(x)
        if shortcut.shape == x.shape:
            x = layers.Add()([shortcut, x])
        else:
            shortcut = layers.Conv1D(filters, kernel_size=1, strides=strides, padding='same')(shortcut)
            x = layers.Add()([shortcut, x])
        x = layers.ReLU()(x)
        return x

    x = residual_depthwise_separable_block(inputs, 16, kernel_size=3, strides=1)
    x = residual_depthwise_separable_block(x, 32, kernel_size=3, strides=2)
    x = residual_depthwise_separable_block(x, 64, kernel_size=3, strides=1)
    x = residual_depthwise_separable_block(x, 128, kernel_size=3, strides=2)
    x = residual_depthwise_separable_block(x, 256, kernel_size=3, strides=1)
    x = residual_depthwise_separable_block(x, 512, kernel_size=3, strides=2)
    x = residual_depthwise_separable_block(x, 1028, kernel_size=3, strides=2)
    x = residual_depthwise_separable_block(x, 2056, kernel_size=3, strides=2)

    # Attention Layer
    x = layers.Attention()([x, x])

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = layers.Dense(86, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)

    return models.Model(inputs=inputs, outputs=x)


# Initialize the model without the output layer
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    base_model = prepare_model_without_output(input_shape=(150, 1))  # Shared base model

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
            best_model_path = '/gpfs/fs7/aafc/phenocart/PhenomicsProjects/UFPSGPSCProject/6_Output/5_Final Results/2024/Best_1D_HeightEstimation_Model.h5'
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
            '1D_HeightEstimation_Results',
            'xlsx'
        )
        results_df.to_excel(results_file_path, index=False)

# At the end of all repetitions, display the best model path
print(f"The best model was saved to {best_model_path}")
