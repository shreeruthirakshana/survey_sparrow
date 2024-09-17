# Generic Churn Prediction System

## Overview

This project is a flexible and robust churn prediction system designed to handle multiple datasets with varying features. The system includes a predictive model, an API for predictions, and interpretability techniques to explain model decisions. It aims to build a scalable machine learning pipeline that can be deployed and extended for various churn prediction datasets.

## Project Structure
generic-churn-prediction/ 
├── data/ │ 
├── churn_dataset1.csv │ 
├── churn_dataset2.csv
│ └── churn_dataset3.csv 
├── notebooks/ 
│ └── churn_analysis.ipynb 
├── src/ 
│ ├── main.py 
│ ├── model.py
│ ├── explanations.py 
│ └── data_utils.py 
├── requirements.txt 
├── README.md └
── .gitignore


## Features

1. **Data Preprocessing & EDA:**
   - **`data_utils.py`**: Contains functions for preprocessing and feature engineering.
   - **`churn_analysis.ipynb`**: Jupyter Notebook for exploratory data analysis and model training.

2. **Modeling:**
   - **`model.py`**: Defines and trains machine learning models, including Random Forest and XGBoost.

3. **Explainability:**
   - **`explanations.py`**: Provides methods for model explainability using techniques like SHAP and LIME.

4. **API Development:**
   - **`main.py`**: FastAPI application that provides endpoints for predictions and model explanations.

## Setup

1. **Create and Activate Virtual Environment:**

   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate

2. **install dependencies:**
pip install -r requirements.txt

3. **run jupyter notebook:**
jupyter notebook notebooks/churn_analysis.ipynb


4. **Fast api server:**
uvicorn src.main:app --reloadu


Usage
**Endpoint:**
POST /predict/: Predict churn based on input features.
POST /explain/global/: Get global model explanations (SHAP).
POST /explain/local/: Get local model explanations (LIME).
POST /explain/surrogate/: Get surrogate model explanations.

**Notes**
Data Files: Ensure that the dataset files are placed in the data/ directory.
Model Training: If no model is found, the system will retrain using the available data.
Dependencies: The requirements.txt file includes necessary libraries for the project.
