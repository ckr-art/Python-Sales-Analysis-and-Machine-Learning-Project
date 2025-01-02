#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

Orders=pd.read_csv('Orders_Dataset.csv')
Sale=pd.read_csv('Sales_Dataset.csv')

df=pd.DataFrame(Orders)
df=pd.DataFrame(Sale)


print(Sale.head())
print(Orders.head())


# In[ ]:





# In[2]:


print("Missing Value in Orders")
print(Orders.isnull().sum())

print("Missing Value in Sale")
print(Sale.isnull().sum())


# In[3]:


Sale.info()


# In[4]:


print("Duplicate records in Orders")
print(Orders.duplicated().sum())

print("Duplicate records in Sale")
print(Sale.duplicated().sum())


# In[29]:


merged_data=pd.merge(Orders,Sale, on='Order ID',how='inner')
print(merged_data.head())
df.to_csv('merged_data.csv')


# In[6]:


print(merged_data.duplicated().sum())


# In[7]:


merged_data['Order Date']=pd.to_datetime(merged_data['Order Date'])
print(merged_data.dtypes)


# In[8]:


print(merged_data.describe())


# In[9]:


import matplotlib.pyplot as plt

sales_trend=merged_data.groupby('Order Date') ['Amount'].sum()

plt.figure(figsize=(10, 6))
sales_trend.plot(title='Sales Trend Over Time', color='blue')
plt.xlabel('Order Date')
plt.ylabel('Total Sales')
plt.grid()
plt.show()


# In[10]:


state_profit=merged_data.groupby('State') ['Profit'].sum().sort_values(ascending=False)

plt.figure(figsize=(12,8))
state_profit.plot(kind='bar', title='State WIse Total Profit', color='green')
plt.xlabel('State')
plt.ylabel('Amount')
plt.show()


# category_sale=merged_data.groupby('Category') ['Amount']
# 
# category_sale.plot(kind='pie', title='Category Total Sale')
# plt.ylabel('')
# plt.show()

# In[11]:


category_sale=merged_data.groupby('Category') ['Amount'].sum()

category_sale.plot(kind='pie', title='Category Total Sale', autopct='%1.2f%%')
plt.ylabel('')
plt.show()


# In[14]:


import seaborn as sns

# Count of each payment mode
payment_mode = merged_data['Payment Mode'].value_counts()

# Plot payment mode popularity
plt.figure(figsize=(10, 6))
sns.barplot(x=payment_mode.index, y=payment_mode.values, palette='viridis')
plt.title('Payment Mode Popularity')
plt.xlabel('Payment Mode')
plt.ylabel('Count')
plt.grid()
plt.show()

Top_state=merged_data.groupby('State') ['Amount'].sum().nlargest(5)

print("The Top 5 State On Sale")
print(Top_state)
# In[17]:


Top_subcategory=merged_data.groupby('Sub-Category')['Profit'].sum().nlargest(1)
print("The Top Sub-Category")
print(Top_subcategory)


# In[19]:


Avg_Qty=merged_data.groupby('Category') ['Quantity'].mean()
print("Average Qty sold per Category")
print(Avg_Qty)


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Encode categorical variables
categorical_cols = ['State', 'City', 'Category', 'Sub-Category', 'PaymentMode']
encoder = LabelEncoder()

for col in categorical_cols:
    merged_data[col] = encoder.fit_transform(merged_data[col])

# Select features and target variable
features = merged_data[['State', 'City', 'Category', 'Sub-Category', 'Quantity', 'PaymentMode']]
target = merged_data['Profit']  # Example: Predicting Profit

# Normalize numerical features
scaler = StandardScaler()
features[['Quantity']] = scaler.fit_transform(features[['Quantity']])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)


# In[ ]:


import joblib

# Save the model
joblib.dump(model, 'profit_prediction_model.pkl')

# To load the model later
# model = joblib.load('profit_prediction_model.pkl')


# In[ ]:


from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the model
model = joblib.load('profit_prediction_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    df = pd.DataFrame(data)
    
    # Preprocess input data (ensure it matches training data preprocessing)
    df['Quantity'] = scaler.transform(df[['Quantity']])
    df[categorical_cols] = df[categorical_cols].apply(encoder.transform)
    
    # Make prediction
    prediction = model.predict(df)
    return jsonify({'predicted_profit': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)


# 
# 

# In[ ]:





# In[ ]:




