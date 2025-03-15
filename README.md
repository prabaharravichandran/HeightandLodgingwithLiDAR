# Precision Phenotyping in Wheat: LiDAR-Based Plant Height and Lodging Estimation Using Unmanned Ground Vehicle

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
![PyPI - Python Version](https://img.shields.io/badge/python-3.12-blue)

Accurate phenotyping of wheat traits supports breeding innovations and strengthens research methodologies in crop sciences. This study focuses on Light Detection and Ranging  (LiDAR)-based evaluation of crop canopy height and lodging in spring wheat using unmanned ground vehicles (UGVs). The University of Saskatchewan Field Phenotyping Systems (UFPS) was deployed across six research stations in Canada during the 2023 and 2024 field seasons. A total of 90 plots, representing 30 historical wheat cultivars, were planted in a Randomized Complete Block Design (RCBD), with LiDAR data collected at selective growth stages at all locations.

Four data processing methods—height distribution vector, aerial projection, orthographic projection, and voxelized spatial grid—were assessed for their ability to represent canopy structure and lodging. For crop canopy height estimation, the aerial projection (95.98%) and orthographic projection (96.05%) methods achieved the highest accuracy, with low Root Mean Squared Error (RMSE) and Mean Absolute Percentage Error (MAPE), approximately 3.7 cm and 3.55%, respectively. In comparison, the voxelized spatial grid and height distribution vector methods showed lower performance, with R-squared values of 88.15% and 84.37% and RMSE values of 6.81 cm and 8 cm, respectively.

For lodging estimation, a progressive fine-tuning approach was used to iteratively categorize lodging severity into 2 to 9 classes. The aerial projection method consistently outperformed other approaches across all metrics and class numbers, achieving the highest accuracy (up to 98.76%), Quadratic Weighted Kappa (QWK, 0.97), and Macro-F1 scores (up to 98.51%), particularly excelling in 2- and 3-class scenarios. The 3D method demonstrated competitive performance, particularly in Macro-F1 and QWK, but slightly lagged behind the aerial projection method. In contrast, the height distribution vector and orthographic projection methods generally exhibited lower performance, especially with higher class numbers (5 and 9), highlighting their limited scalability compared to the aerial projection and voxelized spatial grid methods.

These findings demonstrate the effectiveness of LiDAR-based approaches, particularly aerial projection, for high-throughput phenotyping in spring wheat, offering robust methods to enhance trait estimation for breeding programs.

<div align="center">
  <img src="https://prabahar.s3.ca-central-1.amazonaws.com/static/articles/Phenocart.jpg" alt="Phenocart" width="3000">
  <p><i>Figure 1: UFPS that includes RTK base station and wheeled robot controlled by remote controller.</i></p>
</div>

## Setting up the environment

The environment is built over Python 3.12. Create and activate environment a virtual environment,

```bash
python -m venv .venv # .venv is the name of the environment
. /.venv/bin/activate
```

cd to the source folder and install requirements

```bash
pip install -r requirements.txt
```