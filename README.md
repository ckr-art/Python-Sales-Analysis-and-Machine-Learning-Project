# Python-Sales-Analysis-and-Machine-Learning-Project

This repository contains a complete workflow for analyzing sales data and building a machine learning model to predict profit. It includes data preprocessing, exploratory data analysis (EDA), model training, and deployment.

Data Description

File 1: Order Details

SL NO: Serial number

Order ID: Unique identifier for each order

Order Date: Date of the order

Customer Name: Name of the customer

State: State of the customer

City: City of the customer

File 2: Sales Details

SL NO: Serial number

Order ID: Unique identifier for each order

Amount: Total sales amount

Profit: Profit for the order

Quantity: Quantity of items sold

Category: Product category

Sub-Category: Product sub-category

PaymentMode: Mode of payment


Merged Data

The datasets are merged on the Order ID column to create a comprehensive dataset for analysis and modeling.


Steps to Reproduce

1. Data Analysis

The data_analysis.py script performs the following:

Data loading and inspection

Data cleaning (handling missing values, duplicates, etc.)


Exploratory Data Analysis (EDA), including:

Sales trends over time

Profit by state

Sales by category

Payment mode popularity

2. Machine Learning Model

A regression model is trained to predict Profit using features like State, City, Category, Sub-Category, Quantity, and PaymentMode.

The model is saved as profit_prediction_model.pkl.

3. Model Deployment

A Flask app (app.py) is provided to deploy the model as a REST API.

The API accepts JSON input and returns predicted profit values.

Setup Instructions

Prerequisites

Python 3.8 or later

Install required packages:

pip install -r requirements.txt

Running the Scripts

Data Analysis:

python scripts/data_analysis.py

Flask App:

python deployment/app.py

The app will be available at http://127.0.0.1:5000.

API Usage

Send a POST request to /predict with JSON input. Example:

{
  "State": ["California"],
  "City": ["Los Angeles"],
  "Category": ["Technology"],
  "Sub-Category": ["Phones"],
  "Quantity": [5],
  "PaymentMode": ["Credit Card"]
}

Response:

{
  "predicted_profit": [125.67]
}

Visualizations

The following visualizations are included:

Sales Trend: visualizations/sales_trend.png

Profit by State: visualizations/profit_by_state.png

Category-wise Sales: visualizations/category_sales.png

Payment Mode Popularity: visualizations/payment_mode.png
