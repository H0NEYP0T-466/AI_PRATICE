"""
Pandas Basics: Data Manipulation and Analysis
==============================================

This project demonstrates core Pandas functionality including:
- DataFrame and Series creation
- Data loading and saving
- Data cleaning and preprocessing
- Data analysis and aggregation
- Grouping and merging operations
"""

import pandas as pd
import numpy as np
import os

def create_sample_data():
    """Create sample datasets for demonstration"""
    print("=" * 50)
    print("CREATING SAMPLE DATA")
    print("=" * 50)
    
    # Create a sample sales dataset
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    
    sales_data = {
        'Date': dates,
        'Product': np.random.choice(['Laptop', 'Phone', 'Tablet', 'Desktop', 'Monitor'], 100),
        'Category': np.random.choice(['Electronics', 'Computers', 'Mobile'], 100),
        'Sales_Amount': np.random.normal(1000, 200, 100).round(2),
        'Quantity': np.random.randint(1, 10, 100),
        'Region': np.random.choice(['North', 'South', 'East', 'West'], 100),
        'Customer_Age': np.random.randint(18, 70, 100),
        'Customer_Satisfaction': np.random.uniform(1, 5, 100).round(1)
    }
    
    df = pd.DataFrame(sales_data)
    
    # Add some missing values for cleaning demonstration
    missing_indices = np.random.choice(df.index, 10, replace=False)
    df.loc[missing_indices, 'Customer_Satisfaction'] = np.nan
    
    print("Sample sales dataset created:")
    print(df.head())
    print(f"\nDataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    return df

def basic_dataframe_operations(df):
    """Demonstrate basic DataFrame operations"""
    print("\n" + "=" * 50)
    print("BASIC DATAFRAME OPERATIONS")
    print("=" * 50)
    
    # Basic info about the DataFrame
    print("Dataset Info:")
    print(f"Shape: {df.shape}")
    print(f"Column types:\n{df.dtypes}")
    print(f"Memory usage: {df.memory_usage().sum()} bytes")
    
    # Basic statistics
    print("\nNumerical columns summary:")
    print(df.describe())
    
    # Categorical columns summary
    print("\nCategorical columns info:")
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        print(f"{col}: {df[col].nunique()} unique values")
        print(f"  Most common: {df[col].value_counts().head(3).to_dict()}")
    
    # Missing values
    print(f"\nMissing values:\n{df.isnull().sum()}")
    
    return categorical_cols

def data_selection_and_filtering(df):
    """Demonstrate data selection and filtering techniques"""
    print("\n" + "=" * 50)
    print("DATA SELECTION AND FILTERING")
    print("=" * 50)
    
    # Column selection
    print("Single column selection (Product):")
    print(df['Product'].head())
    
    print("\nMultiple column selection:")
    print(df[['Product', 'Sales_Amount', 'Region']].head())
    
    # Row selection
    print("\nFirst 3 rows:")
    print(df.head(3))
    
    print("\nRows 5-10:")
    print(df.iloc[5:10][['Product', 'Sales_Amount', 'Region']])
    
    # Filtering based on conditions
    high_sales = df[df['Sales_Amount'] > 1200]
    print(f"\nHigh sales (>$1200): {len(high_sales)} records")
    print(high_sales[['Product', 'Sales_Amount', 'Region']].head())
    
    # Multiple conditions
    laptop_north = df[(df['Product'] == 'Laptop') & (df['Region'] == 'North')]
    print(f"\nLaptops sold in North region: {len(laptop_north)} records")
    
    # Using query method
    young_customers = df.query('Customer_Age < 30 and Sales_Amount > 800')
    print(f"\nYoung customers (age < 30) with sales > $800: {len(young_customers)} records")
    
    return high_sales, laptop_north

def data_cleaning_operations(df):
    """Demonstrate data cleaning operations"""
    print("\n" + "=" * 50)
    print("DATA CLEANING OPERATIONS")
    print("=" * 50)
    
    # Create a copy for cleaning
    df_clean = df.copy()
    
    print(f"Original missing values:\n{df_clean.isnull().sum()}")
    
    # Handle missing values
    # Fill missing satisfaction scores with median
    median_satisfaction = df_clean['Customer_Satisfaction'].median()
    df_clean['Customer_Satisfaction'].fillna(median_satisfaction, inplace=True)
    
    print(f"\nAfter filling missing values:\n{df_clean.isnull().sum()}")
    
    # Remove duplicates (if any)
    initial_rows = len(df_clean)
    df_clean.drop_duplicates(inplace=True)
    print(f"\nRemoved {initial_rows - len(df_clean)} duplicate rows")
    
    # Data type conversions
    df_clean['Date'] = pd.to_datetime(df_clean['Date'])
    print(f"\nData types after cleaning:\n{df_clean.dtypes}")
    
    # Create new calculated columns
    df_clean['Revenue'] = df_clean['Sales_Amount'] * df_clean['Quantity']
    df_clean['Month'] = df_clean['Date'].dt.month
    df_clean['Weekday'] = df_clean['Date'].dt.day_name()
    
    print("\nNew calculated columns added:")
    print(df_clean[['Sales_Amount', 'Quantity', 'Revenue', 'Month', 'Weekday']].head())
    
    return df_clean

def data_aggregation_and_grouping(df_clean):
    """Demonstrate data aggregation and grouping"""
    print("\n" + "=" * 50)
    print("DATA AGGREGATION AND GROUPING")
    print("=" * 50)
    
    # Basic aggregations
    print("Overall statistics:")
    print(f"Total Revenue: ${df_clean['Revenue'].sum():,.2f}")
    print(f"Average Sales Amount: ${df_clean['Sales_Amount'].mean():.2f}")
    print(f"Total Quantity Sold: {df_clean['Quantity'].sum()}")
    
    # Group by operations
    print("\nRevenue by Product:")
    product_revenue = df_clean.groupby('Product')['Revenue'].agg(['sum', 'mean', 'count'])
    product_revenue.columns = ['Total_Revenue', 'Avg_Revenue', 'Sales_Count']
    print(product_revenue.sort_values('Total_Revenue', ascending=False))
    
    print("\nSales by Region and Product:")
    region_product = df_clean.groupby(['Region', 'Product'])['Revenue'].sum().unstack(fill_value=0)
    print(region_product)
    
    # Time-based analysis
    print("\nMonthly sales trends:")
    monthly_sales = df_clean.groupby('Month').agg({
        'Revenue': 'sum',
        'Sales_Amount': 'mean',
        'Customer_Satisfaction': 'mean'
    }).round(2)
    print(monthly_sales)
    
    # Custom aggregations
    print("\nCustom aggregations by Region:")
    custom_agg = df_clean.groupby('Region').agg({
        'Revenue': ['sum', 'mean', 'std'],
        'Customer_Age': ['mean', 'min', 'max'],
        'Customer_Satisfaction': ['mean', 'count']
    }).round(2)
    print(custom_agg)
    
    return product_revenue, monthly_sales

def data_manipulation_operations(df_clean):
    """Demonstrate advanced data manipulation"""
    print("\n" + "=" * 50)
    print("DATA MANIPULATION OPERATIONS")
    print("=" * 50)
    
    # Sorting
    print("Top 10 highest revenue transactions:")
    top_sales = df_clean.nlargest(10, 'Revenue')[['Product', 'Revenue', 'Region', 'Customer_Age']]
    print(top_sales)
    
    # Ranking
    df_clean['Revenue_Rank'] = df_clean['Revenue'].rank(ascending=False)
    print("\nTop 5 ranked revenues:")
    print(df_clean.nsmallest(5, 'Revenue_Rank')[['Product', 'Revenue', 'Revenue_Rank']])
    
    # Pivot tables
    print("\nPivot table - Average sales by Product and Region:")
    pivot_table = df_clean.pivot_table(
        values='Sales_Amount',
        index='Product',
        columns='Region',
        aggfunc='mean',
        fill_value=0
    ).round(2)
    print(pivot_table)
    
    # Cross-tabulation
    print("\nCross-tabulation - Product count by Region:")
    crosstab = pd.crosstab(df_clean['Product'], df_clean['Region'])
    print(crosstab)
    
    # Binning continuous data
    df_clean['Age_Group'] = pd.cut(df_clean['Customer_Age'], 
                                   bins=[0, 30, 50, 100], 
                                   labels=['Young', 'Middle', 'Senior'])
    
    print("\nAge group distribution:")
    print(df_clean['Age_Group'].value_counts())
    
    return pivot_table, crosstab

def data_merging_operations():
    """Demonstrate data merging and joining"""
    print("\n" + "=" * 50)
    print("DATA MERGING OPERATIONS")
    print("=" * 50)
    
    # Create sample customer data
    customer_data = {
        'Customer_ID': range(1, 21),
        'Customer_Name': [f'Customer_{i}' for i in range(1, 21)],
        'Membership_Level': np.random.choice(['Bronze', 'Silver', 'Gold'], 20),
        'Join_Date': pd.date_range('2022-01-01', periods=20, freq='15D')
    }
    customers_df = pd.DataFrame(customer_data)
    
    # Create order data
    order_data = {
        'Order_ID': range(1, 31),
        'Customer_ID': np.random.choice(range(1, 21), 30),
        'Order_Amount': np.random.normal(500, 100, 30).round(2),
        'Order_Date': pd.date_range('2023-01-01', periods=30, freq='5D')
    }
    orders_df = pd.DataFrame(order_data)
    
    print("Customer data:")
    print(customers_df.head())
    print("\nOrder data:")
    print(orders_df.head())
    
    # Inner join
    inner_merged = pd.merge(customers_df, orders_df, on='Customer_ID', how='inner')
    print(f"\nInner join result: {len(inner_merged)} records")
    print(inner_merged.head())
    
    # Left join
    left_merged = pd.merge(customers_df, orders_df, on='Customer_ID', how='left')
    print(f"\nLeft join result: {len(left_merged)} records")
    print(f"Customers without orders: {left_merged['Order_ID'].isnull().sum()}")
    
    # Concatenation
    additional_customers = pd.DataFrame({
        'Customer_ID': range(21, 26),
        'Customer_Name': [f'Customer_{i}' for i in range(21, 26)],
        'Membership_Level': ['Platinum'] * 5,
        'Join_Date': pd.date_range('2023-06-01', periods=5, freq='7D')
    })
    
    all_customers = pd.concat([customers_df, additional_customers], ignore_index=True)
    print(f"\nAfter concatenation: {len(all_customers)} customers")
    
    return inner_merged, all_customers

def save_and_load_data(df_clean):
    """Demonstrate saving and loading data"""
    print("\n" + "=" * 50)
    print("SAVE AND LOAD OPERATIONS")
    print("=" * 50)
    
    # Save to CSV
    csv_file = 'sales_data_clean.csv'
    df_clean.to_csv(csv_file, index=False)
    print(f"Data saved to {csv_file}")
    
    # Load from CSV
    loaded_df = pd.read_csv(csv_file)
    print(f"Data loaded from CSV: {loaded_df.shape}")
    
    # Save subset to JSON
    json_file = 'top_products.json'
    top_products = df_clean.groupby('Product')['Revenue'].sum().sort_values(ascending=False).head(3)
    top_products.to_json(json_file)
    print(f"Top products saved to {json_file}")
    
    # Clean up files
    if os.path.exists(csv_file):
        os.remove(csv_file)
    if os.path.exists(json_file):
        os.remove(json_file)
    
    print("Temporary files cleaned up")
    
    return loaded_df

def main():
    """Main function to run all demonstrations"""
    print("Pandas Basics: Comprehensive Data Analysis Demonstration")
    print("========================================================")
    
    # Create sample data
    df = create_sample_data()
    
    # Basic operations
    categorical_cols = basic_dataframe_operations(df)
    
    # Selection and filtering
    high_sales, laptop_north = data_selection_and_filtering(df)
    
    # Data cleaning
    df_clean = data_cleaning_operations(df)
    
    # Aggregation and grouping
    product_revenue, monthly_sales = data_aggregation_and_grouping(df_clean)
    
    # Data manipulation
    pivot_table, crosstab = data_manipulation_operations(df_clean)
    
    # Merging operations
    inner_merged, all_customers = data_merging_operations()
    
    # Save/load operations
    loaded_df = save_and_load_data(df_clean)
    
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print("✅ Sample data creation and exploration")
    print("✅ Basic DataFrame operations and statistics")
    print("✅ Data selection and filtering techniques")
    print("✅ Data cleaning and preprocessing")
    print("✅ Aggregation and grouping operations")
    print("✅ Advanced data manipulation")
    print("✅ Data merging and joining")
    print("✅ Data saving and loading")
    print("\nPandas basics demonstration completed successfully!")

if __name__ == "__main__":
    main()