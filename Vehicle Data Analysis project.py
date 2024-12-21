import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv(r"\Users\malak\Downloads\archive (3)\vehicles.csv")

# Initial Data Overview
print(data.info())
print(data.head())

# Step 1: Drop Irrelevant Columns
columns_to_drop = ['id', 'url', 'region_url', 'VIN', 'image_url', 'description', 
                   'county', 'lat', 'long', 'posting_date', 'size', 'state']
data.drop(columns=columns_to_drop, axis=1, inplace=True)

# Step 2: Handle Missing Values
data = data.dropna(subset=['year', 'odometer', 'manufacturer', 'model'])  # Drop rows with missing critical columns
data.fillna('unknown', inplace=True)  # Fill remaining missing values with 'unknown'

# Step 3: Remove Duplicates
data = data.drop_duplicates()

# Step 4: Simplify Categorical Data
# Simplify 'manufacturer'
manufacturer_counts = data['manufacturer'].value_counts()
data['manufacturer'] = data['manufacturer'].apply(lambda x: x if str(x) in manufacturer_counts[:30] else 'others')

# Simplify 'model'
model_counts = data['model'].value_counts()
data['model'] = data['model'].apply(lambda x: x if str(x) in model_counts[:50] else 'others')

# Drop 'region' column as it is too granular
data.drop('region', axis=1, inplace=True)

# Step 5: Remove Outliers
# For 'price'
price_q1 = data['price'].quantile(0.25)
price_q3 = data['price'].quantile(0.75)
price_iqr = price_q3 - price_q1
price_upper_limit = price_q3 + 1.5 * price_iqr
price_lower_limit = data['price'].quantile(0.15)
data = data[(data['price'] < price_upper_limit) & (data['price'] > price_lower_limit)]

# For 'odometer'
odometer_q1 = data['odometer'].quantile(0.25)
odometer_q3 = data['odometer'].quantile(0.75)
odometer_iqr = odometer_q3 - odometer_q1
odometer_upper_limit = odometer_q3 + 1.5 * odometer_iqr
odometer_lower_limit = data['odometer'].quantile(0.05)
data = data[(data['odometer'] < odometer_upper_limit) & (data['odometer'] > odometer_lower_limit)]

# Reset Index
data.reset_index(drop=True, inplace=True)

# Final Dataset Overview
print(data.shape)
print(data.info())

# Visualize Price Distribution
sns.boxplot(data=data, x='price')
plt.xscale('log')
plt.title('Price Distribution (Log Scale)')
plt.show()

# Question 1 - What are the top 5 most common car manufacturers?
top_manufacturers = data['manufacturer'].value_counts().head(5)
print("Top 5 manufacturers:\n", top_manufacturers)

# Question 2 - What is the average price of cars by manufacturer?
avg_price_by_manufacturer = data.groupby('manufacturer')['price'].mean().sort_values(ascending=False)
print("Average price by manufacturer:\n", avg_price_by_manufacturer)

# Question 3 - What is the relationship between mileage (odometer) and price?
sns.scatterplot(data=data, x='odometer', y='price', alpha=0.5)
plt.title('Odometer vs. Price')
plt.xlabel('Mileage (Odometer)')
plt.ylabel('Price')
plt.show()

# Question 4 - How does the condition of the car affect its price?
avg_price_by_condition = data.groupby('condition')['price'].mean().sort_values(ascending=False)
print("Average price by condition:\n", avg_price_by_condition)

sns.boxplot(data=data, x='condition', y='price')
plt.yscale('log')
plt.title('Price by Condition')
plt.show()

# Question 5 - Which fuel type is the most common?
fuel_type_counts = data['fuel'].value_counts()
print("Fuel type distribution:\n", fuel_type_counts)

sns.barplot(x=fuel_type_counts.index, y=fuel_type_counts.values)
plt.title('Fuel Type Distribution')
plt.xlabel('Fuel Type')
plt.ylabel('Count')
plt.show()

# Question 6 - How does transmission type affect price?
avg_price_by_transmission = data.groupby('transmission')['price'].mean().sort_values(ascending=False)
print("Average price by transmission type:\n", avg_price_by_transmission)

sns.boxplot(data=data, x='transmission', y='price')
plt.yscale('log')
plt.title('Price by Transmission Type')
plt.show()

# Question 8 - What are the top 5 models with the highest average price?
avg_price_by_model = data.groupby('model')['price'].mean().sort_values(ascending=False).head(5)
print("Top 5 models with highest average price:\n", avg_price_by_model)

