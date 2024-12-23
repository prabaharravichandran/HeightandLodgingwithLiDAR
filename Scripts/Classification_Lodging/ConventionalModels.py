import sys

sys.path.append('/gpfs/fs7/aafc/phenocart/PhenomicsProjects/cleanqc/')

import pymongo
from gridfs import GridFS
import pandas as pd
from io import BytesIO
from urllib.parse import quote_plus
import numpy as np
import json
import tensorflow as tf

tf.random.set_seed(1)
import plotly.io as pio

# Set Plotly renderer to SVG
pio.renderers.default = 'svg'


def height_extraction(lidar_raw):
    # Extract z values (heights) from LiDAR data
    z = lidar_raw[:, 2]
    height = (z - np.min(z)) * 100

    return height


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
x_train, y_train = [], []

# Process documents and prepare training data

for document in documents:
    if 'lidar_id' in document and variable in document and not pd.isna(document[variable]) and document[
        variable] not in (None, '', [], {}):
        try:
            gridout = fs.get(document['lidar_id'])
            lidar_raw = np.load(BytesIO(gridout.read()))
            resized_grid = height_extraction(lidar_raw)
            print(f"{document['location']}: {document['plot']}")
            x_train.append(resized_grid)
            y_train.append(document[variable])
        except Exception as e:
            print(f"Error processing document ID {document['_id']}: {e}")
    else:
        print(f"Document ID {document['_id']} is missing 'lidar_id' or {variable} data.")

# %% Initialize an empty list to store the results
actual_height = y_train

# Define the percentiles range and step
percentiles = np.arange(80, 100.1, 0.05)
end_percentile = 1
# Initialize results list
results = []

# Loop through the percentile range
for start_percentile in percentiles[:-1]:

    print(start_percentile)
    lidar_height = []
    # Calculate the lidar heights for the current percentile range
    for arr in x_train:
        # Calculate the specified percentiles
        p_start = np.percentile(arr, start_percentile)
        p_end = np.percentile(arr, end_percentile)

        # Calculate the difference
        difference = p_start - p_end
        lidar_height.append(difference)

    # Calculate Pearson correlation coefficient
    correlation_matrix = np.corrcoef(lidar_height, actual_height)
    correlation = correlation_matrix[0, 1]  # Extract the correlation coefficient
    print(correlation)

    # Save results
    results.append({
        "Start Percentile": start_percentile,
        "End Percentile": end_percentile,
        "Pearson Correlation": correlation
    })

# Convert results to a DataFrame
df_results = pd.DataFrame(results)

# Save results to an Excel file
output_path = "/gpfs/fs7/aafc/phenocart/PhenomicsProjects/UFPSGPSCProject/6_Output/pearson_correlation_percentiles.xlsx"
df_results.to_excel(output_path, index=False)

# %% Define the percentage range and step
percentages = np.arange(0.05, 20.05, 0.05)  # Top 0.05% to 20% in 0.05% increments
bottom_percent = 1  # Bottom 1% for comparison

# Initialize results list
percentage_system_results = []

# Loop through the percentage range
for top_percent in percentages:
    print(top_percent)
    lidar_height = []
    # Calculate the lidar heights for the current percentage system
    for arr in x_train:
        # Sort the array to get the top and bottom values directly
        sorted_arr = np.sort(arr)
        n_points = len(sorted_arr)

        # Calculate indices for top and bottom percentages
        top_index_start = int(n_points * (1 - (top_percent / 100)))
        bottom_index_end = int(n_points * (bottom_percent / 100))

        # Calculate the mean of top and bottom percentages
        top_mean = sorted_arr[top_index_start:].mean()
        bottom_mean = sorted_arr[:bottom_index_end].mean()

        # Calculate the difference
        difference = top_mean - bottom_mean
        lidar_height.append(difference)

    # Calculate Pearson correlation coefficient
    correlation_matrix = np.corrcoef(lidar_height, y_train)
    correlation = correlation_matrix[0, 1]  # Extract the correlation coefficient
    print(correlation)

    # Save results
    percentage_system_results.append({
        "Top Percentage": top_percent,
        "Bottom Percentage": bottom_percent,
        "Pearson Correlation": correlation
    })
# Convert results to a DataFrame
df_percentage_system_results = pd.DataFrame(percentage_system_results)

# Save results to an Excel file
output_path_percentage = "/gpfs/fs7/aafc/phenocart/PhenomicsProjects/UFPSGPSCProject/6_Output/percentage_system_correlation.xlsx"
df_percentage_system_results.to_excel(output_path_percentage, index=False)

# %% linear regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Calculate lidar height as the 99.6th percentile
lidar_height_996 = [np.percentile(arr, 99.6) for arr in x_train]

# Reshape data for linear regression
lidar_height_996 = np.array(lidar_height_996).reshape(-1, 1)
actual_height = np.array(y_train).reshape(-1, 1)

# Fit a linear regression model
linear_model = LinearRegression()
linear_model.fit(lidar_height_996, actual_height)

# Predict actual height using the model
predicted_height = linear_model.predict(lidar_height_996)

# Calculate R^2 score
r2 = r2_score(actual_height, predicted_height)

# Calculate RMSE (Root Mean Squared Error)
rmse = mean_squared_error(actual_height, predicted_height, squared=False)

# Display results with RMSE included
{
    "Linear Model Coefficients": linear_model.coef_.flatten(),
    "Linear Model Intercept": linear_model.intercept_.flatten(),
    "R^2 Score": r2,
    "RMSE": rmse
}
# %%with poly

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd

# Store results
results = []

# Calculate lidar height as the 99.6th percentile
lidar_height_996 = [np.percentile(arr, 99.6) for arr in x_train]

# Reshape data for linear regression
lidar_height_996 = np.array(lidar_height_996).reshape(-1, 1)
actual_height = np.array(y_train).reshape(-1, 1)

for degree in range(1, 11):
    print(f"Polynomial Degree: {degree}")

    # Transform features to include polynomial terms
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    lidar_height_poly = poly.fit_transform(lidar_height_996)

    # Fit a polynomial regression model
    linear_model_poly = LinearRegression()
    linear_model_poly.fit(lidar_height_poly, actual_height)

    # Predict and calculate R^2 score and RMSE
    predicted_height_poly = linear_model_poly.predict(lidar_height_poly)
    r2_poly = r2_score(actual_height, predicted_height_poly)
    rmse_poly = mean_squared_error(actual_height, predicted_height_poly, squared=False)

    # Save results
    result = {
        "Polynomial Degree": degree,
        "Linear Model Coefficients": linear_model_poly.coef_,
        "Linear Model Intercept": linear_model_poly.intercept_,
        "R^2 Score": r2_poly,
        "RMSE": rmse_poly
    }
    results.append(result)

# Convert results to a DataFrame for better visualization
df_results = pd.DataFrame(results)

# Print results
print(df_results)

# Optionally save to Excel
output_path = "/gpfs/fs7/aafc/phenocart/PhenomicsProjects/UFPSGPSCProject/6_Output/polynomial_regression_results_with_rmse.xlsx"
df_results.to_excel(output_path, index=False)

# %%plot
import plotly.graph_objects as go

# Create a Plotly figure for R^2 Score vs Polynomial Degree
fig = go.Figure()

# Add a line plot for R^2 Score
fig.add_trace(go.Scatter(
    x=df_results['Polynomial Degree'],
    y=df_results['R^2 Score'] * 100,
    mode='lines+markers',
    name='R^2 Score',
    line=dict(width=2),
    marker=dict(size=8),
    marker_color="#aa239a"
))

# Update layout for better visualization
fig.update_layout(
    title='R<sup>2</sup> (%) vs Polynomial Degree',
    xaxis_title='Polynomial Degree',
    yaxis_title='R<sup>2</sup> (%)',
    template='plotly_white',
    width=800,
    height=600
)

svg_file_path = "/gpfs/fs7/aafc/phenocart/PhenomicsProjects/UFPSGPSCProject/6_Output/R2ScorevsPolynomialDegree.svg"

fig.write_image(svg_file_path, format="svg")

# %% Linear regression with top 1%

# Calculate lidar height as the mean of the top 1%
lidar_height_1percent = [np.mean(np.sort(arr)[-int(0.01 * len(arr)):]) for arr in x_train]

# Reshape data for linear regression
lidar_height_1percent = np.array(lidar_height_1percent).reshape(-1, 1)
actual_height = np.array(y_train).reshape(-1, 1)

# Fit a linear regression model
linear_model = LinearRegression()
linear_model.fit(lidar_height_1percent, actual_height)

# Predict actual height using the model
predicted_height = linear_model.predict(lidar_height_1percent)

# Calculate R^2 score and RMSE
r2 = r2_score(actual_height, predicted_height)
rmse = mean_squared_error(actual_height, predicted_height, squared=False)

# Display results with RMSE included
linear_results = {
    "Linear Model Coefficients": linear_model.coef_.flatten(),
    "Linear Model Intercept": linear_model.intercept_.flatten(),
    "R^2 Score": r2,
    "RMSE": rmse
}
print(linear_results)

# %% Polynomial regression with top 1%

# Store results
results = []

for degree in range(1, 11):
    print(f"Polynomial Degree: {degree}")

    # Transform features to include polynomial terms
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    lidar_height_poly = poly.fit_transform(lidar_height_1percent)

    # Fit a polynomial regression model
    linear_model_poly = LinearRegression()
    linear_model_poly.fit(lidar_height_poly, actual_height)

    # Predict and calculate R^2 score and RMSE
    predicted_height_poly = linear_model_poly.predict(lidar_height_poly)
    r2_poly = r2_score(actual_height, predicted_height_poly)
    rmse_poly = mean_squared_error(actual_height, predicted_height_poly, squared=False)

    # Save results
    result = {
        "Polynomial Degree": degree,
        "Linear Model Coefficients": linear_model_poly.coef_,
        "Linear Model Intercept": linear_model_poly.intercept_,
        "R^2 Score": r2_poly,
        "RMSE": rmse_poly
    }
    results.append(result)

# Convert results to a DataFrame for better visualization
df_results = pd.DataFrame(results)

# Print results
print(df_results)

# Optionally save to Excel
output_path = "/gpfs/fs7/aafc/phenocart/PhenomicsProjects/UFPSGPSCProject/6_Output/polynomial_regression_results_with_rmse_top_1_percent.xlsx"
df_results.to_excel(output_path, index=False)

# %% Plot R^2 Score vs Polynomial Degree

# Create a Plotly figure for R^2 Score vs Polynomial Degree
fig = go.Figure()

# Add a line plot for R^2 Score
fig.add_trace(go.Scatter(
    x=df_results['Polynomial Degree'],
    y=df_results['R^2 Score'] * 100,
    mode='lines+markers',
    name='R^2 Score',
    line=dict(width=2),
    marker=dict(size=8),
    marker_color="#aa239a"
))

# Update layout for better visualization
fig.update_layout(
    title='R<sup>2</sup> vs Polynomial Degree (Top 1% Fraction)',
    xaxis_title='Polynomial Degree',
    yaxis_title='R<sup>2</sup> (%)',
    template='plotly_white',
    width=800,
    height=600
)

svg_file_path = "/gpfs/fs7/aafc/phenocart/PhenomicsProjects/UFPSGPSCProject/6_Output/R2ScorevsPolynomialDegree_top_1_percent.svg"

fig.write_image(svg_file_path, format="svg")

# %% Load the data from the Excel file

file_path = "/gpfs/fs7/aafc/phenocart/PhenomicsProjects/UFPSGPSCProject/6_Output/percentage_system_correlation.xlsx"  # Correct file path for the provided file
df = pd.read_excel(file_path)

# Create a line plot for Top Percentage vs Pearson Correlation
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df['Top Percentage'],
    y=df['Pearson Correlation'],
    mode='lines+markers',
    name='Pearson Correlation',
    line=dict(width=2),
    marker=dict(size=4),
    marker_color="#aa239a"
))

# Update layout for better visualization
fig.update_layout(
    title='Pearson Correlation vs Top Fraction (%)',
    xaxis_title='Top Fraction (%)',
    yaxis_title='Pearson Correlation',
    template='plotly_white',
    width=800,
    height=400
)

svg_file_path = "/gpfs/fs7/aafc/phenocart/PhenomicsProjects/UFPSGPSCProject/6_Output/TopFraction.svg"
fig.write_image(svg_file_path, format="svg")

# %% Load the data from the Excel file

file_path = "/gpfs/fs7/aafc/phenocart/PhenomicsProjects/UFPSGPSCProject/6_Output/pearson_correlation_percentiles.xlsx"  # Correct file path for the provided file
df = pd.read_excel(file_path)

# Create a line plot for Top Percentage vs Pearson Correlation
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df['Start Percentile'],
    y=df['Pearson Correlation'],
    mode='lines+markers',
    name='Pearson Correlation',
    line=dict(width=2),
    marker=dict(size=4),
    marker_color="#aa239a"
))

# Update layout for better visualization
fig.update_layout(
    title='Pearson Correlation vs Start Percentile',
    xaxis_title='Percentile',
    yaxis_title='Pearson Correlation',
    template='plotly_white',
    width=800,
    height=400
)

svg_file_path = "/gpfs/fs7/aafc/phenocart/PhenomicsProjects/UFPSGPSCProject/6_Output/Percentile.svg"
fig.write_image(svg_file_path, format="svg")
