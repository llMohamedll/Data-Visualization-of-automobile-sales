"""
Project 04: Data Visualization with Matplotlib, Seaborn, and Folium
====================================================================
This script demonstrates various data visualization techniques including
line plots, bar charts, pie charts, scatter plots, box plots, and 
interactive maps using Folium.

Original notebook: DV0101EN-Final-Assign-Part1-v1.jupyterlite.ipynb

Author: Data Analysis Portfolio
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
try:
    import folium
except ImportError:
    folium = None
    print('Warning: folium not installed. Install with: pip install folium')
import requests
import io

# ================================================================================
# DATA LOADING
# ================================================================================

def load_automobile_data():
    """Load automobile sales data from URL."""
    url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/d51iMGfp_t0QpO30Lym-dw/automobile-sales.csv"
    
    print(f"Loading data from: {url}")
    try:
        response = requests.get(url)
        response.raise_for_status()
        df = pd.read_csv(io.StringIO(response.text))
        print("Data loaded successfully!")
    except Exception as e:
        print(f"Failed to load from URL: {e}")
        print("Creating sample data for demonstration...")
        # Create sample data
        years = list(range(1980, 2024)) * 4
        np.random.seed(42)
        df = pd.DataFrame({
            'Year': years,
            'Automobile_Sales': np.random.randint(500, 1000, len(years)),
            'Recession': np.random.randint(0, 2, len(years)),
            'Vehicle_Type': np.random.choice(['Sedan', 'SUV', 'Truck', 'Coupe'], len(years)),
            'Advertising_Expenditure': np.random.uniform(1000, 5000, len(years)),
            'Unemployment_Rate': np.random.uniform(3, 15, len(years)),
        })
    
    return df

# ================================================================================
# VISUALIZATION FUNCTIONS
# ================================================================================

def plot_automobile_sales_trend(df):
    """Plot average automobile sales by year."""
    print("\n[1] Creating automobile sales trend plot...")
    
    df_line = df.groupby('Year')['Automobile_Sales'].mean().reset_index()
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Simple line plot
    axes[0].plot(df_line['Year'], df_line['Automobile_Sales'], marker='o', linewidth=2)
    axes[0].set_xlabel('Year')
    axes[0].set_ylabel('Automobile Sales')
    axes[0].set_title('Average Automobile Sales Fluctuate Year to Year')
    axes[0].grid(True, alpha=0.3)
    
    # Line plot with recession markers
    axes[1].plot(df_line['Year'], df_line['Automobile_Sales'], marker='o', linewidth=2)
    axes[1].set_xlabel('Year')
    axes[1].set_ylabel('Automobile Sales')
    axes[1].set_title('Automobile Sales During Recession')
    axes[1].set_xticks(range(1980, 2024, 2))
    axes[1].tick_params(axis='x', rotation=75)
    axes[1].grid(True, alpha=0.3)
    
    # Add recession annotations
    recession_years = [1982, 2008]  # Major recessions
    for yr in recession_years:
        if yr in df_line['Year'].values:
            y_val = df_line[df_line['Year'] == yr]['Automobile_Sales'].values[0]
            axes[1].annotate(f'{yr} Recession', xy=(yr, y_val), 
                           xytext=(yr, y_val + 50),
                           arrowprops=dict(arrowstyle='->', color='red'),
                           fontsize=10, color='red')
    
    plt.tight_layout()
    plt.savefig('automobile_sales_trend.png', dpi=150, bbox_inches='tight')
    print("Saved automobile_sales_trend.png")
    plt.close()

def plot_vehicle_type_analysis(df):
    """Analyze sales by vehicle type."""
    print("\n[2] Creating vehicle type analysis...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Bar chart: Average sales by vehicle type
    vt_sales = df.groupby('Vehicle_Type')['Automobile_Sales'].mean().sort_values()
    axes[0, 0].barh(vt_sales.index, vt_sales.values, color='steelblue')
    axes[0, 0].set_xlabel('Average Sales')
    axes[0, 0].set_title('Average Sales by Vehicle Type')
    
    # Box plot: Sales distribution by vehicle type
    vehicle_types = df['Vehicle_Type'].unique()
    sales_data = [df[df['Vehicle_Type'] == vt]['Automobile_Sales'] for vt in vehicle_types]
    axes[0, 1].boxplot(sales_data, tick_labels=vehicle_types)
    axes[0, 1].set_ylabel('Sales')
    axes[0, 1].set_title('Sales Distribution by Vehicle Type')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Pie chart: Sales proportion by vehicle type
    vt_total = df.groupby('Vehicle_Type')['Automobile_Sales'].sum()
    axes[1, 0].pie(vt_total.values, labels=vt_total.index, autopct='%1.1f%%', startangle=90)
    axes[1, 0].set_title('Total Sales Proportion by Vehicle Type')
    
    # Sales during recession vs non-recession by vehicle type
    recession_sales = df[df['Recession'] == 1].groupby('Vehicle_Type')['Automobile_Sales'].mean()
    normal_sales = df[df['Recession'] == 0].groupby('Vehicle_Type')['Automobile_Sales'].mean()
    
    x = np.arange(len(vehicle_types))
    width = 0.35
    axes[1, 1].bar(x - width/2, normal_sales.values, width, label='Normal', color='green', alpha=0.7)
    axes[1, 1].bar(x + width/2, recession_sales.values, width, label='Recession', color='red', alpha=0.7)
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(vehicle_types)
    axes[1, 1].set_ylabel('Average Sales')
    axes[1, 1].set_title('Sales by Vehicle Type: Recession vs Normal')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('vehicle_type_analysis.png', dpi=150, bbox_inches='tight')
    print("Saved vehicle_type_analysis.png")
    plt.close()

def plot_advertising_analysis(df):
    """Analyze advertising expenditure impact."""
    print("\n[3] Creating advertising analysis...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Scatter plot: Advertising vs Sales
    axes[0].scatter(df['Advertising_Expenditure'], df['Automobile_Sales'], alpha=0.5)
    axes[0].set_xlabel('Advertising Expenditure')
    axes[0].set_ylabel('Automobile Sales')
    axes[0].set_title('Advertising Expenditure vs Sales')
    
    # Add trend line
    z = np.polyfit(df['Advertising_Expenditure'], df['Automobile_Sales'], 1)
    p = np.poly1d(z)
    axes[0].plot(df['Advertising_Expenditure'], p(df['Advertising_Expenditure']), "r--", alpha=0.8)
    
    # Scatter plot: Unemployment vs Sales
    axes[1].scatter(df['Unemployment_Rate'], df['Automobile_Sales'], alpha=0.5, color='orange')
    axes[1].set_xlabel('Unemployment Rate (%)')
    axes[1].set_ylabel('Automobile Sales')
    axes[1].set_title('Unemployment Rate vs Sales')
    
    # Add trend line
    z = np.polyfit(df['Unemployment_Rate'], df['Automobile_Sales'], 1)
    p = np.poly1d(z)
    axes[1].plot(df['Unemployment_Rate'], p(df['Unemployment_Rate']), "r--", alpha=0.8)
    
    plt.tight_layout()
    plt.savefig('advertising_analysis.png', dpi=150, bbox_inches='tight')
    print("Saved advertising_analysis.png")
    plt.close()

def create_folium_map():
    """Create an interactive Folium map."""
    print("\n[4] Creating Folium map...")
    if folium is None:
        print("  Folium not installed, skipping map creation")
        return

    
    # Create a map centered on a location (e.g., Chicago)
    m = folium.Map(location=[41.8781, -87.6298], zoom_start=10)
    
    # Add some sample markers
    locations = [
        [41.8781, -87.6298, "Chicago Downtown"],
        [41.9742, -87.9073, "O'Hare Airport"],
        [41.7868, -87.5967, "Midway Airport"],
        [42.0565, -87.6743, "Evanston"],
    ]
    
    for lat, lon, name in locations:
        folium.Marker(
            location=[lat, lon],
            popup=name,
            icon=folium.Icon(color='blue', icon='info-sign')
        ).add_to(m)
    
    # Add a circle marker
    folium.Circle(
        location=[41.8781, -87.6298],
        radius=5000,
        color='red',
        fill=True,
        fill_color='red',
        fill_opacity=0.2,
        popup='Chicago Metro Area'
    ).add_to(m)
    
    # Save map
    m.save('automobile_sales_map.html')
    print("Saved automobile_sales_map.html")
    
    return m

# ================================================================================
# MAIN EXECUTION
# ================================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("PROJECT 04: DATA VISUALIZATION (MATPLOTLIB, SEABORN, FOLIUM)")
    print("=" * 60)
    
    # Load data
    df = load_automobile_data()
    print(f"\nData shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nSample data:\n{df.head()}")
    
    # Create visualizations
    plot_automobile_sales_trend(df)
    plot_vehicle_type_analysis(df)
    plot_advertising_analysis(df)
    create_folium_map()
    
    print("\n" + "=" * 60)
    print("ALL VISUALIZATIONS COMPLETE")
    print("=" * 60)
    print("\nGenerated files:")
    print("  - automobile_sales_trend.png")
    print("  - vehicle_type_analysis.png")
    print("  - advertising_analysis.png")
    print("  - automobile_sales_map.html")
