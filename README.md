# Car-Purchase-Amount-Predictor
A machine learning application that predicts potential car purchase amounts based on customer demographics and financial information. This project includes both a data analysis pipeline and a user-friendly GUI interface for making predictions.

# Overview
This application helps car dealerships and sales teams predict how much a customer might spend on a car purchase based on their profile data. The system uses various regression models to analyze the relationships between customer attributes (age, income, net worth, etc.) and their car purchasing behavior.

# Features

Data preprocessing pipeline with encoding detection, missing value handling, and feature scaling
Feature engineering to create meaningful derived attributes like debt-to-income ratio
Multiple regression models with cross-validation and anti-overfitting techniques
Model evaluation using RMSE, MAE, and R² metrics
Feature importance analysis to understand key factors affecting purchase decisions
Interactive GUI application for easy predictions without coding knowledge
Comprehensive documentation and usage examples

# Installation
Prerequisites

Python 3.8+
Required packages:

pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
tkinter (usually included with Python)



# Setup

Clone this repository:
git clone https://github.com/username/car-purchase-predictor.git
cd car-purchase-predictor

Install required packages:
pip install -r requirements.txt

Place your dataset in the project directory (or use the sample data)

# Usage
Data Analysis and Model Training
Run the main analysis script to process data and train models:
python car_purchase_analysis.py
This will:

Load and preprocess the dataset
Perform exploratory data analysis
Create engineered features
Train and evaluate multiple regression models
Save the best performing model as car_purchase_model.pkl

# GUI Prediction Interface
After training the model, launch the GUI application:
python car_purchase_predictor_gui.py
The interface allows you to:

Enter customer information (age, salary, debt, net worth, etc.)
Get instant predictions of potential car purchase amounts
Reset inputs for multiple predictions

# Dataset Information
The application works with customer data containing the following fields:

customer name - Customer's name
customer e-mail - Customer's email address
country - Customer's country
gender - Customer's gender
age - Customer's age
annual Salary - Customer's annual income
credit card debt - Customer's credit card debt
net worth - Customer's total net worth
car purchase amount - Target variable to predict

# Model Performance
The application tries multiple regression models:

Ridge Regression
Lasso Regression
ElasticNet
Random Forest
XGBoost

# Performance metrics include:

Root Mean Squared Error (RMSE)
Mean Absolute Error (MAE)
R² Score
Cross-validation scores

Implementation Details
Preprocessing Pipeline

Automatic encoding detection for CSV files
Separate pipelines for numerical and categorical features
Standard scaling for numerical features
One-hot encoding for categorical features

Feature Engineering

Debt-to-income ratio
Net worth-to-salary ratio
Age grouping

Anti-Overfitting Techniques

Cross-validation
Regularization (L1, L2, Elastic Net)
Conservative tree-based model parameters
Train/test performance comparison

# GUI Implementation

Tkinter-based interface
Input validation
Clear results display
Simple, intuitive design

# Files Description

car_purchase_analysis.py - Main analysis script
car_purchase_predictor_gui.py - Standalone GUI application
car_purchase_model.pkl - Trained prediction model
requirements.txt - Required Python packages
data/ - Directory for datasets
screenshots/ - Sample screenshots of the application

# Customization
Using Your Own Dataset
Replace the default file path in the analysis script:
pythondf = load_data('path/to/your/dataset.csv')
Ensure your dataset has similar columns or adjust the feature names accordingly.
Modifying the Models
To try different algorithms or parameters, modify the models dictionary in the analysis script:
pythonmodels = {
    "Your New Model": YourModelClass(parameters),
    # Other models...
}
# Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

# Acknowledgments

This project was created as a demonstration of practical machine learning and GUI development
Inspired by the need for data-driven decision making in the automotive sales industry
