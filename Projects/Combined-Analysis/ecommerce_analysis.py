"""
Combined Data Analysis Project: NumPy + Pandas + Matplotlib
===========================================================

This project demonstrates a complete data analysis pipeline using all three libraries:
- NumPy: For numerical computations and array operations
- Pandas: For data manipulation, cleaning, and analysis
- Matplotlib: For comprehensive data visualization

Real-world scenario: E-commerce Sales Analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os

# Set plotting style
plt.style.use('default')
plt.rcParams['figure.figsize'] = (12, 8)

def create_plots_directory():
    """Create plots directory if it doesn't exist"""
    plots_dir = 'plots'
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    return plots_dir

def generate_ecommerce_dataset():
    """Generate a realistic e-commerce dataset using NumPy"""
    print("=" * 60)
    print("STEP 1: DATA GENERATION WITH NUMPY")
    print("=" * 60)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Define parameters
    n_customers = 1000
    n_products = 50
    n_transactions = 5000
    
    # Generate customer data
    customer_ages = np.random.normal(35, 12, n_customers).astype(int)
    customer_ages = np.clip(customer_ages, 18, 80)  # Ensure realistic age range
    
    # Generate product data
    product_categories = ['Electronics', 'Clothing', 'Home & Garden', 'Sports', 'Books', 'Beauty']
    product_base_prices = np.array([299, 59, 89, 129, 19, 39])  # Base prices by category
    
    # Generate transaction data using NumPy arrays
    transaction_dates = pd.date_range('2023-01-01', '2023-12-31', freq='H')
    selected_dates = np.random.choice(len(transaction_dates), n_transactions, replace=True)
    
    # Create arrays for transaction components
    customer_ids = np.random.randint(1, n_customers + 1, n_transactions)
    product_ids = np.random.randint(1, n_products + 1, n_transactions)
    
    # Generate realistic quantities (most transactions are 1-3 items)
    quantities = np.random.choice([1, 2, 3, 4, 5], n_transactions, p=[0.5, 0.25, 0.15, 0.07, 0.03])
    
    # Generate category assignments for products
    product_categories_assigned = np.random.choice(product_categories, n_products)
    
    # Generate prices with some variation
    category_indices = np.array([product_categories.index(cat) for cat in product_categories_assigned])
    base_prices = product_base_prices[category_indices]
    price_variations = np.random.normal(1, 0.2, n_products)
    price_variations = np.clip(price_variations, 0.5, 2.0)
    product_prices = base_prices * price_variations
    
    print(f"✅ Generated data for {n_customers} customers")
    print(f"✅ Generated data for {n_products} products across {len(product_categories)} categories")
    print(f"✅ Generated {n_transactions} transactions")
    print(f"📊 Customer age statistics: Mean={np.mean(customer_ages):.1f}, Std={np.std(customer_ages):.1f}")
    print(f"💰 Price range: ${np.min(product_prices):.2f} - ${np.max(product_prices):.2f}")
    
    return {
        'customer_ages': customer_ages,
        'customer_ids': customer_ids,
        'product_ids': product_ids,
        'product_categories': product_categories_assigned,
        'product_prices': product_prices,
        'quantities': quantities,
        'transaction_dates': transaction_dates[selected_dates],
        'n_customers': n_customers,
        'n_products': n_products
    }

def create_pandas_dataframes(data_dict):
    """Create and process Pandas DataFrames from NumPy arrays"""
    print("\n" + "=" * 60)
    print("STEP 2: DATA PROCESSING WITH PANDAS")
    print("=" * 60)
    
    # Create customers DataFrame
    customers_df = pd.DataFrame({
        'customer_id': range(1, data_dict['n_customers'] + 1),
        'age': data_dict['customer_ages'],
    })
    
    # Add calculated customer segments based on age using NumPy operations
    age_percentiles = np.percentile(customers_df['age'], [33, 67])
    customers_df['age_segment'] = pd.cut(customers_df['age'], 
                                        bins=[0, age_percentiles[0], age_percentiles[1], 100],
                                        labels=['Young', 'Middle', 'Senior'])
    
    # Create products DataFrame
    products_df = pd.DataFrame({
        'product_id': range(1, data_dict['n_products'] + 1),
        'category': data_dict['product_categories'],
        'price': data_dict['product_prices']
    })
    
    # Create transactions DataFrame
    transactions_df = pd.DataFrame({
        'transaction_id': range(1, len(data_dict['customer_ids']) + 1),
        'customer_id': data_dict['customer_ids'],
        'product_id': data_dict['product_ids'],
        'quantity': data_dict['quantities'],
        'transaction_date': data_dict['transaction_dates']
    })
    
    # Merge DataFrames to create comprehensive dataset
    print("🔄 Merging DataFrames...")
    full_data = transactions_df.merge(customers_df, on='customer_id') \
                              .merge(products_df, on='product_id')
    
    # Calculate revenue using NumPy operations
    full_data['revenue'] = full_data['price'] * full_data['quantity']
    
    # Add time-based features using Pandas datetime functionality
    full_data['month'] = full_data['transaction_date'].dt.month
    full_data['day_of_week'] = full_data['transaction_date'].dt.day_name()
    full_data['hour'] = full_data['transaction_date'].dt.hour
    full_data['is_weekend'] = full_data['transaction_date'].dt.weekday >= 5
    
    # Add seasonal information
    def get_season(month):
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Fall'
    
    full_data['season'] = full_data['month'].apply(get_season)
    
    print(f"📊 Final dataset shape: {full_data.shape}")
    print(f"🗓️  Date range: {full_data['transaction_date'].min()} to {full_data['transaction_date'].max()}")
    print(f"💰 Total revenue: ${full_data['revenue'].sum():,.2f}")
    print(f"🛍️  Average order value: ${full_data['revenue'].mean():.2f}")
    
    return full_data, customers_df, products_df, transactions_df

def data_analysis_with_pandas(full_data):
    """Perform comprehensive data analysis using Pandas"""
    print("\n" + "=" * 60)
    print("STEP 3: COMPREHENSIVE DATA ANALYSIS")
    print("=" * 60)
    
    # Basic statistics
    print("📈 REVENUE STATISTICS:")
    revenue_stats = full_data['revenue'].describe()
    print(revenue_stats)
    
    # Category analysis
    print("\n🏷️  CATEGORY PERFORMANCE:")
    category_analysis = full_data.groupby('category').agg({
        'revenue': ['sum', 'mean', 'count'],
        'quantity': 'sum',
        'customer_id': 'nunique'
    }).round(2)
    category_analysis.columns = ['Total_Revenue', 'Avg_Order_Value', 'Total_Orders', 
                                'Units_Sold', 'Unique_Customers']
    category_analysis = category_analysis.sort_values('Total_Revenue', ascending=False)
    print(category_analysis)
    
    # Customer segment analysis
    print("\n👥 CUSTOMER SEGMENT ANALYSIS:")
    segment_analysis = full_data.groupby('age_segment').agg({
        'revenue': ['sum', 'mean'],
        'transaction_id': 'count',
        'customer_id': 'nunique'
    }).round(2)
    segment_analysis.columns = ['Total_Revenue', 'Avg_Order_Value', 'Total_Orders', 'Unique_Customers']
    print(segment_analysis)
    
    # Time-based analysis
    print("\n📅 MONTHLY TRENDS:")
    monthly_trends = full_data.groupby('month').agg({
        'revenue': 'sum',
        'transaction_id': 'count',
        'customer_id': 'nunique'
    }).round(2)
    monthly_trends.columns = ['Revenue', 'Orders', 'Customers']
    print(monthly_trends.head(12))
    
    # Day of week analysis
    print("\n📆 DAY OF WEEK PERFORMANCE:")
    dow_analysis = full_data.groupby('day_of_week')['revenue'].agg(['sum', 'mean', 'count']).round(2)
    dow_analysis.columns = ['Total_Revenue', 'Avg_Order_Value', 'Order_Count']
    # Reorder by actual day order
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    dow_analysis = dow_analysis.reindex(day_order)
    print(dow_analysis)
    
    # Seasonal analysis using NumPy calculations
    print("\n🌍 SEASONAL ANALYSIS:")
    seasonal_analysis = full_data.groupby('season').agg({
        'revenue': ['sum', 'mean'],
        'quantity': 'sum'
    }).round(2)
    seasonal_analysis.columns = ['Total_Revenue', 'Avg_Order_Value', 'Units_Sold']
    print(seasonal_analysis)
    
    # Customer behavior analysis
    print("\n🎯 CUSTOMER BEHAVIOR INSIGHTS:")
    customer_behavior = full_data.groupby('customer_id').agg({
        'revenue': ['sum', 'count'],
        'transaction_date': ['min', 'max']
    })
    customer_behavior.columns = ['Total_Spent', 'Order_Count', 'First_Order', 'Last_Order']
    customer_behavior['Customer_Lifetime_Days'] = (customer_behavior['Last_Order'] - 
                                                   customer_behavior['First_Order']).dt.days
    
    print(f"💰 Top 5 customers by spending:")
    top_customers = customer_behavior.nlargest(5, 'Total_Spent')[['Total_Spent', 'Order_Count']]
    print(top_customers)
    
    print(f"\n🔄 Most frequent customers:")
    frequent_customers = customer_behavior.nlargest(5, 'Order_Count')[['Total_Spent', 'Order_Count']]
    print(frequent_customers)
    
    # Calculate customer metrics using NumPy
    avg_customer_value = np.mean(customer_behavior['Total_Spent'])
    customer_retention_days = np.mean(customer_behavior['Customer_Lifetime_Days'])
    repeat_customer_rate = np.sum(customer_behavior['Order_Count'] > 1) / len(customer_behavior)
    
    print(f"\n📊 KEY METRICS:")
    print(f"   • Average Customer Lifetime Value: ${avg_customer_value:.2f}")
    print(f"   • Average Customer Retention: {customer_retention_days:.1f} days")
    print(f"   • Repeat Customer Rate: {repeat_customer_rate:.1%}")
    
    return {
        'category_analysis': category_analysis,
        'segment_analysis': segment_analysis,
        'monthly_trends': monthly_trends,
        'dow_analysis': dow_analysis,
        'seasonal_analysis': seasonal_analysis,
        'customer_behavior': customer_behavior,
        'key_metrics': {
            'avg_customer_value': avg_customer_value,
            'retention_days': customer_retention_days,
            'repeat_rate': repeat_customer_rate
        }
    }

def create_comprehensive_visualizations(full_data, analysis_results, plots_dir):
    """Create comprehensive visualizations using Matplotlib"""
    print("\n" + "=" * 60)
    print("STEP 4: DATA VISUALIZATION WITH MATPLOTLIB")
    print("=" * 60)
    
    # Set color palette
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    # 1. Revenue by Category
    plt.figure(figsize=(15, 10))
    
    # Category revenue pie chart
    plt.subplot(2, 3, 1)
    category_revenue = analysis_results['category_analysis']['Total_Revenue']
    plt.pie(category_revenue.values, labels=category_revenue.index, autopct='%1.1f%%', 
            colors=colors[:len(category_revenue)], startangle=90)
    plt.title('Revenue Distribution by Category', fontsize=12, fontweight='bold')
    
    # Monthly trends line plot
    plt.subplot(2, 3, 2)
    monthly_data = analysis_results['monthly_trends']
    plt.plot(monthly_data.index, monthly_data['Revenue'], marker='o', linewidth=2, color='#1f77b4')
    plt.title('Monthly Revenue Trends', fontsize=12, fontweight='bold')
    plt.xlabel('Month')
    plt.ylabel('Revenue ($)')
    plt.grid(True, alpha=0.3)
    
    # Day of week bar chart
    plt.subplot(2, 3, 3)
    dow_data = analysis_results['dow_analysis']
    bars = plt.bar(range(len(dow_data)), dow_data['Total_Revenue'], color=colors[1])
    plt.title('Revenue by Day of Week', fontsize=12, fontweight='bold')
    plt.xlabel('Day of Week')
    plt.ylabel('Total Revenue ($)')
    plt.xticks(range(len(dow_data)), dow_data.index, rotation=45)
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.annotate(f'${height:,.0f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
    
    # Customer segment analysis
    plt.subplot(2, 3, 4)
    segment_data = analysis_results['segment_analysis']
    plt.bar(segment_data.index, segment_data['Total_Revenue'], color=colors[2])
    plt.title('Revenue by Customer Segment', fontsize=12, fontweight='bold')
    plt.xlabel('Age Segment')
    plt.ylabel('Total Revenue ($)')
    
    # Seasonal analysis
    plt.subplot(2, 3, 5)
    seasonal_data = analysis_results['seasonal_analysis']
    plt.bar(seasonal_data.index, seasonal_data['Total_Revenue'], color=colors[3])
    plt.title('Revenue by Season', fontsize=12, fontweight='bold')
    plt.xlabel('Season')
    plt.ylabel('Total Revenue ($)')
    
    # Revenue distribution histogram
    plt.subplot(2, 3, 6)
    plt.hist(full_data['revenue'], bins=50, alpha=0.7, color=colors[4], edgecolor='black')
    plt.axvline(np.mean(full_data['revenue']), color='red', linestyle='--', 
                label=f'Mean: ${np.mean(full_data["revenue"]):.2f}')
    plt.title('Revenue Distribution', fontsize=12, fontweight='bold')
    plt.xlabel('Revenue ($)')
    plt.ylabel('Frequency')
    plt.legend()
    
    plt.tight_layout()
    plot_path = os.path.join(plots_dir, 'comprehensive_analysis.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Advanced Time Series Analysis
    plt.figure(figsize=(15, 8))
    
    # Daily revenue trends
    daily_revenue = full_data.groupby(full_data['transaction_date'].dt.date)['revenue'].sum()
    
    plt.subplot(2, 2, 1)
    plt.plot(daily_revenue.index, daily_revenue.values, alpha=0.7, color='blue')
    # Add moving average using NumPy
    window_size = 7
    moving_avg = np.convolve(daily_revenue.values, np.ones(window_size)/window_size, mode='valid')
    plt.plot(daily_revenue.index[window_size-1:], moving_avg, color='red', linewidth=2, 
             label=f'{window_size}-day Moving Average')
    plt.title('Daily Revenue with Moving Average', fontsize=12, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Revenue ($)')
    plt.legend()
    plt.xticks(rotation=45)
    
    # Hourly patterns
    plt.subplot(2, 2, 2)
    hourly_revenue = full_data.groupby('hour')['revenue'].mean()
    plt.plot(hourly_revenue.index, hourly_revenue.values, marker='o', linewidth=2)
    plt.title('Average Revenue by Hour', fontsize=12, fontweight='bold')
    plt.xlabel('Hour of Day')
    plt.ylabel('Average Revenue ($)')
    plt.grid(True, alpha=0.3)
    
    # Weekend vs Weekday comparison
    plt.subplot(2, 2, 3)
    weekend_data = full_data.groupby('is_weekend')['revenue'].agg(['sum', 'mean'])
    weekend_labels = ['Weekday', 'Weekend']
    x_pos = np.arange(len(weekend_labels))
    
    bars1 = plt.bar(x_pos - 0.2, weekend_data['sum'], 0.4, label='Total Revenue', alpha=0.8)
    plt.ylabel('Total Revenue ($)', color='blue')
    plt.tick_params(axis='y', labelcolor='blue')
    
    # Create second y-axis for average
    ax2 = plt.gca().twinx()
    bars2 = ax2.bar(x_pos + 0.2, weekend_data['mean'], 0.4, label='Average Revenue', 
                    color='orange', alpha=0.8)
    ax2.set_ylabel('Average Revenue ($)', color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')
    
    plt.title('Weekend vs Weekday Performance', fontsize=12, fontweight='bold')
    plt.xticks(x_pos, weekend_labels)
    
    # Customer age vs spending scatter plot
    plt.subplot(2, 2, 4)
    customer_age_spending = full_data.groupby('customer_id').agg({
        'age': 'first',
        'revenue': 'sum'
    })
    
    plt.scatter(customer_age_spending['age'], customer_age_spending['revenue'], 
                alpha=0.6, color='purple', s=30)
    
    # Add trend line using NumPy
    z = np.polyfit(customer_age_spending['age'], customer_age_spending['revenue'], 1)
    p = np.poly1d(z)
    plt.plot(customer_age_spending['age'], p(customer_age_spending['age']), 
             "r--", alpha=0.8, linewidth=2)
    
    plt.title('Customer Age vs Total Spending', fontsize=12, fontweight='bold')
    plt.xlabel('Customer Age')
    plt.ylabel('Total Spending ($)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(plots_dir, 'time_series_analysis.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. Customer Analysis Heatmap
    plt.figure(figsize=(12, 8))
    
    # Create customer behavior matrix
    customer_matrix = full_data.pivot_table(
        values='revenue',
        index='age_segment',
        columns='category',
        aggfunc='mean',
        fill_value=0
    )
    
    # Create heatmap
    im = plt.imshow(customer_matrix.values, cmap='YlOrRd', aspect='auto')
    
    # Add labels
    plt.xticks(range(len(customer_matrix.columns)), customer_matrix.columns, rotation=45)
    plt.yticks(range(len(customer_matrix.index)), customer_matrix.index)
    plt.xlabel('Product Category', fontsize=12)
    plt.ylabel('Customer Age Segment', fontsize=12)
    plt.title('Average Order Value by Segment and Category', fontsize=14, fontweight='bold')
    
    # Add value annotations
    for i in range(len(customer_matrix.index)):
        for j in range(len(customer_matrix.columns)):
            text = plt.text(j, i, f'${customer_matrix.values[i, j]:.0f}',
                           ha="center", va="center", color="black", fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im)
    cbar.set_label('Average Order Value ($)', fontsize=12)
    
    plt.tight_layout()
    plot_path = os.path.join(plots_dir, 'customer_heatmap.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"📊 Saved comprehensive_analysis.png")
    print(f"📊 Saved time_series_analysis.png") 
    print(f"📊 Saved customer_heatmap.png")
    
    return daily_revenue, customer_matrix

def generate_insights_report(full_data, analysis_results):
    """Generate final insights using all three libraries"""
    print("\n" + "=" * 60)
    print("STEP 5: AUTOMATED INSIGHTS GENERATION")
    print("=" * 60)
    
    # Use NumPy for statistical calculations
    revenue_data = full_data['revenue'].values
    
    # Key statistical insights using NumPy
    total_revenue = np.sum(revenue_data)
    avg_revenue = np.mean(revenue_data)
    median_revenue = np.median(revenue_data)
    revenue_std = np.std(revenue_data)
    revenue_percentiles = np.percentile(revenue_data, [25, 75])
    
    # Business insights using Pandas analysis
    top_category = analysis_results['category_analysis']['Total_Revenue'].idxmax()
    top_category_revenue = analysis_results['category_analysis']['Total_Revenue'].max()
    
    best_month = analysis_results['monthly_trends']['Revenue'].idxmax()
    best_month_revenue = analysis_results['monthly_trends']['Revenue'].max()
    
    best_day = analysis_results['dow_analysis']['Total_Revenue'].idxmax()
    
    # Customer insights
    total_customers = full_data['customer_id'].nunique()
    avg_orders_per_customer = len(full_data) / total_customers
    
    print("🎯 KEY BUSINESS INSIGHTS:")
    print("=" * 40)
    print(f"💰 Financial Performance:")
    print(f"   • Total Revenue: ${total_revenue:,.2f}")
    print(f"   • Average Order Value: ${avg_revenue:.2f}")
    print(f"   • Median Order Value: ${median_revenue:.2f}")
    print(f"   • Revenue Standard Deviation: ${revenue_std:.2f}")
    print(f"   • 25th-75th Percentile Range: ${revenue_percentiles[0]:.2f} - ${revenue_percentiles[1]:.2f}")
    
    print(f"\n🏆 Top Performers:")
    print(f"   • Best Category: {top_category} (${top_category_revenue:,.2f})")
    print(f"   • Best Month: Month {best_month} (${best_month_revenue:,.2f})")
    print(f"   • Best Day: {best_day}")
    
    print(f"\n👥 Customer Metrics:")
    print(f"   • Total Customers: {total_customers:,}")
    print(f"   • Average Orders per Customer: {avg_orders_per_customer:.1f}")
    print(f"   • Customer Lifetime Value: ${analysis_results['key_metrics']['avg_customer_value']:.2f}")
    print(f"   • Repeat Customer Rate: {analysis_results['key_metrics']['repeat_rate']:.1%}")
    
    # Seasonality insights using NumPy correlation
    seasonal_revenue = analysis_results['seasonal_analysis']['Total_Revenue']
    seasonal_ranking = seasonal_revenue.rank(ascending=False)
    
    print(f"\n🌍 Seasonal Insights:")
    for season, rank in seasonal_ranking.items():
        revenue = seasonal_revenue[season]
        print(f"   • {season}: ${revenue:,.2f} (Rank #{int(rank)})")
    
    # Growth trend analysis using NumPy
    monthly_revenues = analysis_results['monthly_trends']['Revenue'].values
    if len(monthly_revenues) >= 2:
        growth_rates = np.diff(monthly_revenues) / monthly_revenues[:-1] * 100
        avg_growth_rate = np.mean(growth_rates)
        print(f"\n📈 Growth Analysis:")
        print(f"   • Average Monthly Growth Rate: {avg_growth_rate:.1f}%")
        if avg_growth_rate > 0:
            print(f"   • Trend: Positive growth trajectory ✅")
        else:
            print(f"   • Trend: Declining revenue trend ⚠️")
    
    print(f"\n💡 Recommendations:")
    print(f"   • Focus marketing on {top_category} category")
    print(f"   • Optimize operations for {best_day} peak demand")
    print(f"   • Develop retention strategies (current rate: {analysis_results['key_metrics']['repeat_rate']:.1%})")
    if analysis_results['segment_analysis']['Total_Revenue'].idxmax():
        top_segment = analysis_results['segment_analysis']['Total_Revenue'].idxmax()
        print(f"   • Target {top_segment} customer segment for premium offerings")

def main():
    """Main function to run the complete analysis pipeline"""
    print("🚀 COMPREHENSIVE E-COMMERCE DATA ANALYSIS")
    print("==========================================")
    print("Using NumPy + Pandas + Matplotlib for End-to-End Analysis")
    print()
    
    # Create plots directory
    plots_dir = create_plots_directory()
    
    # Step 1: Data Generation with NumPy
    data_dict = generate_ecommerce_dataset()
    
    # Step 2: Data Processing with Pandas
    full_data, customers_df, products_df, transactions_df = create_pandas_dataframes(data_dict)
    
    # Step 3: Data Analysis with Pandas
    analysis_results = data_analysis_with_pandas(full_data)
    
    # Step 4: Data Visualization with Matplotlib
    daily_revenue, customer_matrix = create_comprehensive_visualizations(
        full_data, analysis_results, plots_dir)
    
    # Step 5: Insights Generation using all three libraries
    generate_insights_report(full_data, analysis_results)
    
    print("\n" + "=" * 60)
    print("✅ ANALYSIS COMPLETE!")
    print("=" * 60)
    print("📊 NumPy: Used for numerical computations, statistical analysis, and data generation")
    print("🐼 Pandas: Used for data manipulation, cleaning, grouping, and analysis")
    print("📈 Matplotlib: Used for comprehensive visualizations and charts")
    print(f"💾 All visualizations saved to: {plots_dir}/")
    print("\n🎯 This project demonstrates a complete real-world data analysis workflow!")

if __name__ == "__main__":
    main()