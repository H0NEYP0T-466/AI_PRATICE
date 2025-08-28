"""
Matplotlib Basics: Data Visualization and Plotting
===================================================

This project demonstrates comprehensive Matplotlib functionality including:
- Basic plotting (line, scatter, bar, histogram)
- Plot customization and styling
- Subplots and multiple plots
- Statistical visualizations
- Advanced plot types
- Plot saving and export
"""

import matplotlib.pyplot as plt
import numpy as np
import os

# Set the plotting style
plt.style.use('default')

def create_plots_directory():
    """Create plots directory if it doesn't exist"""
    plots_dir = 'plots'
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    return plots_dir

def basic_line_plots(plots_dir):
    """Demonstrate basic line plotting"""
    print("=" * 50)
    print("BASIC LINE PLOTS")
    print("=" * 50)
    
    # Generate sample data
    x = np.linspace(0, 2*np.pi, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)
    y3 = np.sin(x) * np.cos(x)
    
    # Create line plot
    plt.figure(figsize=(10, 6))
    plt.plot(x, y1, label='sin(x)', linewidth=2, color='blue')
    plt.plot(x, y2, label='cos(x)', linewidth=2, color='red', linestyle='--')
    plt.plot(x, y3, label='sin(x)cos(x)', linewidth=2, color='green', linestyle=':')
    
    plt.title('Trigonometric Functions', fontsize=16, fontweight='bold')
    plt.xlabel('x (radians)', fontsize=12)
    plt.ylabel('y values', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 2*np.pi)
    
    # Save the plot
    plot_path = os.path.join(plots_dir, 'line_plots.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Line plot saved to: {plot_path}")
    return x, y1, y2, y3

def scatter_plots(plots_dir):
    """Demonstrate scatter plotting"""
    print("\n" + "=" * 50)
    print("SCATTER PLOTS")
    print("=" * 50)
    
    # Generate sample data
    np.random.seed(42)
    n_points = 200
    x = np.random.normal(0, 1, n_points)
    y = 2 * x + np.random.normal(0, 0.5, n_points)
    colors = np.random.rand(n_points)
    sizes = np.random.randint(20, 200, n_points)
    
    # Create scatter plot
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(x, y, c=colors, s=sizes, alpha=0.6, cmap='viridis')
    
    plt.title('Scatter Plot with Color and Size Variation', fontsize=16, fontweight='bold')
    plt.xlabel('X values', fontsize=12)
    plt.ylabel('Y values', fontsize=12)
    plt.colorbar(scatter, label='Color Scale')
    plt.grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    plt.plot(x, p(x), "r--", alpha=0.8, linewidth=2, label=f'Trend: y={z[0]:.2f}x+{z[1]:.2f}')
    plt.legend()
    
    # Save the plot
    plot_path = os.path.join(plots_dir, 'scatter_plot.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Scatter plot saved to: {plot_path}")
    return x, y, colors, sizes

def bar_charts(plots_dir):
    """Demonstrate bar charts"""
    print("\n" + "=" * 50)
    print("BAR CHARTS")
    print("=" * 50)
    
    # Sample data
    categories = ['Product A', 'Product B', 'Product C', 'Product D', 'Product E']
    sales_2022 = [120, 95, 180, 150, 200]
    sales_2023 = [135, 110, 165, 175, 220]
    
    x_pos = np.arange(len(categories))
    width = 0.35
    
    # Create bar chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Grouped bar chart
    bars1 = ax1.bar(x_pos - width/2, sales_2022, width, label='2022', color='skyblue', alpha=0.8)
    bars2 = ax1.bar(x_pos + width/2, sales_2023, width, label='2023', color='lightcoral', alpha=0.8)
    
    ax1.set_title('Sales Comparison by Product', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Products', fontsize=12)
    ax1.set_ylabel('Sales (units)', fontsize=12)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(categories, rotation=45)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.annotate(f'{height}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    for bar in bars2:
        height = bar.get_height()
        ax1.annotate(f'{height}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    # Horizontal bar chart
    ax2.barh(categories, sales_2023, color='lightgreen', alpha=0.8)
    ax2.set_title('2023 Sales (Horizontal)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Sales (units)', fontsize=12)
    ax2.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(plots_dir, 'bar_charts.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Bar charts saved to: {plot_path}")
    return categories, sales_2022, sales_2023

def histograms_and_distributions(plots_dir):
    """Demonstrate histograms and distribution plots"""
    print("\n" + "=" * 50)
    print("HISTOGRAMS AND DISTRIBUTIONS")
    print("=" * 50)
    
    # Generate sample data
    np.random.seed(42)
    normal_data = np.random.normal(100, 15, 1000)
    uniform_data = np.random.uniform(50, 150, 1000)
    exponential_data = np.random.exponential(20, 1000)
    
    # Create histogram plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Normal distribution
    ax1.hist(normal_data, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.set_title('Normal Distribution', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Values')
    ax1.set_ylabel('Frequency')
    ax1.axvline(np.mean(normal_data), color='red', linestyle='--', label=f'Mean: {np.mean(normal_data):.1f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Uniform distribution
    ax2.hist(uniform_data, bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
    ax2.set_title('Uniform Distribution', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Values')
    ax2.set_ylabel('Frequency')
    ax2.axvline(np.mean(uniform_data), color='red', linestyle='--', label=f'Mean: {np.mean(uniform_data):.1f}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Exponential distribution
    ax3.hist(exponential_data, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
    ax3.set_title('Exponential Distribution', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Values')
    ax3.set_ylabel('Frequency')
    ax3.axvline(np.mean(exponential_data), color='red', linestyle='--', label=f'Mean: {np.mean(exponential_data):.1f}')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Overlapping histograms
    ax4.hist(normal_data, bins=25, alpha=0.5, label='Normal', color='blue')
    ax4.hist(uniform_data, bins=25, alpha=0.5, label='Uniform', color='red')
    ax4.set_title('Overlapping Distributions', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Values')
    ax4.set_ylabel('Frequency')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(plots_dir, 'histograms.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Histograms saved to: {plot_path}")
    return normal_data, uniform_data, exponential_data

def pie_charts_and_donut(plots_dir):
    """Demonstrate pie charts and donut charts"""
    print("\n" + "=" * 50)
    print("PIE CHARTS AND DONUT CHARTS")
    print("=" * 50)
    
    # Sample data
    labels = ['Desktop', 'Mobile', 'Tablet', 'Smart TV', 'Other']
    sizes = [45, 30, 15, 7, 3]
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ff99cc']
    explode = (0.1, 0, 0, 0, 0)  # explode first slice
    
    # Create pie and donut charts
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Pie chart
    wedges, texts, autotexts = ax1.pie(sizes, explode=explode, labels=labels, colors=colors,
                                       autopct='%1.1f%%', shadow=True, startangle=90)
    ax1.set_title('Device Usage Distribution', fontsize=14, fontweight='bold')
    
    # Beautify pie chart
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    # Donut chart
    wedges2, texts2, autotexts2 = ax2.pie(sizes, labels=labels, colors=colors,
                                          autopct='%1.1f%%', startangle=90, 
                                          wedgeprops=dict(width=0.5))
    ax2.set_title('Device Usage (Donut Chart)', fontsize=14, fontweight='bold')
    
    # Add center circle for donut effect
    centre_circle = plt.Circle((0, 0), 0.70, fc='white')
    ax2.add_artist(centre_circle)
    
    # Beautify donut chart
    for autotext in autotexts2:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(plots_dir, 'pie_charts.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Pie charts saved to: {plot_path}")
    return labels, sizes

def box_plots_and_violin_plots(plots_dir):
    """Demonstrate box plots and violin plots"""
    print("\n" + "=" * 50)
    print("BOX PLOTS AND VIOLIN PLOTS")
    print("=" * 50)
    
    # Generate sample data
    np.random.seed(42)
    data1 = np.random.normal(100, 10, 200)
    data2 = np.random.normal(90, 15, 200)
    data3 = np.random.normal(110, 8, 200)
    data4 = np.random.normal(95, 20, 200)
    
    data = [data1, data2, data3, data4]
    labels = ['Group A', 'Group B', 'Group C', 'Group D']
    
    # Create box and violin plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Box plot
    box_plot = ax1.boxplot(data, labels=labels, patch_artist=True)
    ax1.set_title('Box Plot Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Values')
    ax1.grid(True, alpha=0.3)
    
    # Color the boxes
    colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow']
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
    
    # Violin plot
    violin_plot = ax2.violinplot(data, positions=range(1, len(data)+1), showmeans=True, showmedians=True)
    ax2.set_title('Violin Plot Comparison', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Values')
    ax2.set_xticks(range(1, len(data)+1))
    ax2.set_xticklabels(labels)
    ax2.grid(True, alpha=0.3)
    
    # Color the violins
    for i, pc in enumerate(violin_plot['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.7)
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(plots_dir, 'box_violin_plots.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Box and violin plots saved to: {plot_path}")
    return data, labels

def heatmaps_and_correlation(plots_dir):
    """Demonstrate heatmaps and correlation plots"""
    print("\n" + "=" * 50)
    print("HEATMAPS AND CORRELATION PLOTS")
    print("=" * 50)
    
    # Generate correlation matrix data
    np.random.seed(42)
    variables = ['Temperature', 'Humidity', 'Pressure', 'Wind Speed', 'Rainfall']
    n_vars = len(variables)
    
    # Create a correlation matrix
    correlation_matrix = np.random.rand(n_vars, n_vars)
    correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2  # Make symmetric
    np.fill_diagonal(correlation_matrix, 1)  # Diagonal should be 1
    
    # Create heatmap
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Correlation heatmap
    im1 = ax1.imshow(correlation_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
    ax1.set_title('Correlation Matrix Heatmap', fontsize=14, fontweight='bold')
    ax1.set_xticks(range(n_vars))
    ax1.set_yticks(range(n_vars))
    ax1.set_xticklabels(variables, rotation=45)
    ax1.set_yticklabels(variables)
    
    # Add correlation values to cells
    for i in range(n_vars):
        for j in range(n_vars):
            text = ax1.text(j, i, f'{correlation_matrix[i, j]:.2f}',
                           ha="center", va="center", color="black", fontweight='bold')
    
    # Add colorbar
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label('Correlation Coefficient')
    
    # Random data heatmap
    random_data = np.random.rand(10, 12)
    im2 = ax2.imshow(random_data, cmap='viridis', aspect='auto')
    ax2.set_title('Random Data Heatmap', fontsize=14, fontweight='bold')
    ax2.set_xlabel('X axis')
    ax2.set_ylabel('Y axis')
    
    # Add colorbar
    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label('Values')
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(plots_dir, 'heatmaps.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Heatmaps saved to: {plot_path}")
    return correlation_matrix, variables

def subplots_demonstration(plots_dir):
    """Demonstrate subplot arrangements"""
    print("\n" + "=" * 50)
    print("SUBPLOTS DEMONSTRATION")
    print("=" * 50)
    
    # Generate data
    x = np.linspace(0, 10, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)
    y3 = np.tan(x)
    y4 = np.exp(-x/5) * np.sin(x)
    
    # Create complex subplot arrangement
    fig = plt.figure(figsize=(15, 10))
    
    # Main plot (top half)
    ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=3)
    ax1.plot(x, y1, 'b-', label='sin(x)', linewidth=2)
    ax1.plot(x, y2, 'r--', label='cos(x)', linewidth=2)
    ax1.set_title('Main Plot: Trigonometric Functions', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Bottom left
    ax2 = plt.subplot2grid((3, 3), (1, 0), colspan=1)
    ax2.plot(x, y3, 'g-', linewidth=2)
    ax2.set_title('tan(x)', fontsize=12)
    ax2.set_ylim(-5, 5)
    ax2.grid(True, alpha=0.3)
    
    # Bottom middle
    ax3 = plt.subplot2grid((3, 3), (1, 1), colspan=1)
    ax3.plot(x, y4, 'm-', linewidth=2)
    ax3.set_title('Damped sine', fontsize=12)
    ax3.grid(True, alpha=0.3)
    
    # Bottom right
    ax4 = plt.subplot2grid((3, 3), (1, 2), colspan=1)
    ax4.scatter(x[::10], y1[::10], c='red', s=50)
    ax4.set_title('Scatter points', fontsize=12)
    ax4.grid(True, alpha=0.3)
    
    # Bottom spanning plot
    ax5 = plt.subplot2grid((3, 3), (2, 0), colspan=3)
    ax5.hist(np.random.normal(0, 1, 1000), bins=30, alpha=0.7, color='orange')
    ax5.set_title('Random Distribution', fontsize=12)
    ax5.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(plots_dir, 'subplots_demo.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Subplots demonstration saved to: {plot_path}")
    return x, y1, y2, y3, y4

def plot_customization_demo(plots_dir):
    """Demonstrate advanced plot customization"""
    print("\n" + "=" * 50)
    print("PLOT CUSTOMIZATION DEMONSTRATION")
    print("=" * 50)
    
    # Generate data
    x = np.linspace(0, 4*np.pi, 1000)
    y = np.sin(x) * np.exp(-x/10)
    
    # Create highly customized plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Main plot line
    line = ax.plot(x, y, linewidth=3, color='#2E86AB', label='Damped Sine Wave')
    
    # Fill area under curve
    ax.fill_between(x, y, alpha=0.3, color='#A23B72')
    
    # Customize title and labels
    ax.set_title('Advanced Plot Customization Demo', 
                fontsize=20, fontweight='bold', pad=20, 
                color='#2E86AB')
    
    ax.set_xlabel('Time (s)', fontsize=14, fontweight='bold', color='#333333')
    ax.set_ylabel('Amplitude', fontsize=14, fontweight='bold', color='#333333')
    
    # Customize ticks
    ax.tick_params(axis='both', which='major', labelsize=12, 
                   colors='#333333', width=2, length=6)
    
    # Add grid with custom styling
    ax.grid(True, linestyle='--', alpha=0.7, color='#CCCCCC', linewidth=1)
    
    # Customize spines
    for spine in ax.spines.values():
        spine.set_color('#333333')
        spine.set_linewidth(2)
    
    # Add annotations
    max_idx = np.argmax(y)
    ax.annotate(f'Maximum: ({x[max_idx]:.2f}, {y[max_idx]:.2f})',
                xy=(x[max_idx], y[max_idx]), 
                xytext=(x[max_idx]+2, y[max_idx]+0.2),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    # Add text box
    textstr = 'Features:\\n• Custom colors\\n• Fill area\\n• Annotations\\n• Grid styling'
    props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    # Add legend
    ax.legend(loc='upper right', fontsize=12, frameon=True, 
              fancybox=True, shadow=True)
    
    # Set background color
    ax.set_facecolor('#F8F9FA')
    fig.patch.set_facecolor('white')
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(plots_dir, 'customization_demo.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print(f"Customization demo saved to: {plot_path}")
    return x, y

def main():
    """Main function to run all demonstrations"""
    print("Matplotlib Basics: Comprehensive Visualization Demonstration")
    print("===========================================================")
    
    # Create plots directory
    plots_dir = create_plots_directory()
    print(f"Plots will be saved to: {plots_dir}/")
    
    # Run all demonstrations
    x, y1, y2, y3 = basic_line_plots(plots_dir)
    x_scatter, y_scatter, colors, sizes = scatter_plots(plots_dir)
    categories, sales_2022, sales_2023 = bar_charts(plots_dir)
    normal_data, uniform_data, exponential_data = histograms_and_distributions(plots_dir)
    labels, sizes_pie = pie_charts_and_donut(plots_dir)
    data_box, labels_box = box_plots_and_violin_plots(plots_dir)
    correlation_matrix, variables = heatmaps_and_correlation(plots_dir)
    x_sub, y1_sub, y2_sub, y3_sub, y4_sub = subplots_demonstration(plots_dir)
    x_custom, y_custom = plot_customization_demo(plots_dir)
    
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print("✅ Basic line plots with multiple series")
    print("✅ Scatter plots with color and size variation")
    print("✅ Bar charts (grouped and horizontal)")
    print("✅ Histograms and distribution comparisons")
    print("✅ Pie charts and donut charts")
    print("✅ Box plots and violin plots")
    print("✅ Heatmaps and correlation matrices")
    print("✅ Complex subplot arrangements")
    print("✅ Advanced plot customization")
    print(f"\nAll plots saved to: {plots_dir}/")
    print("Matplotlib basics demonstration completed successfully!")

if __name__ == "__main__":
    main()