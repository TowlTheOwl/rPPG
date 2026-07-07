import numpy as np
import matplotlib.pyplot as plt

def create_bland_altman(data1, data2):
    # Convert data into NumPy arrays
    data1 = np.asarray(data1)
    data2 = np.asarray(data2)
    
    # Calculate means and differences
    mean = np.mean([data1, data2], axis=0)
    diff = data1 - data2
    
    # Calculate average bias and standard deviation of differences
    md = np.mean(diff)
    sd = np.std(diff, ddof=1) # Sample standard deviation
    
    # Calculate 95% limits of agreement (LoA)
    upper_loa = md + 1.96 * sd
    lower_loa = md - 1.96 * sd
    
    # Initialize the plot
    plt.figure(figsize=(8, 6))
    plt.scatter(mean, diff, color='blue', alpha=0.6, edgecolors='k')
    
    # Add the horizontal lines for Bias and Limits of Agreement
    plt.axhline(md, color='red', linestyle='-', linewidth=1.5, label=f'Bias ({md:.2f})')
    plt.axhline(upper_loa, color='gray', linestyle='--', linewidth=1.5, label=f'+1.96 SD ({upper_loa:.2f})')
    plt.axhline(lower_loa, color='gray', linestyle='--', linewidth=1.5, label=f'-1.96 SD ({lower_loa:.2f})')
    
    # Offset text styling for the lines
    x_text_pos = max(mean) * 0.95
    plt.text(x_text_pos, md, f'Bias: {md:.2f}', va='bottom', ha='right', color='red', fontweight='bold')
    plt.text(x_text_pos, upper_loa, f'+1.96 SD: {upper_loa:.2f}', va='bottom', ha='right', color='gray')
    plt.text(x_text_pos, lower_loa, f'-1.96 SD: {lower_loa:.2f}', va='top', ha='right', color='gray')
    
    # Labels, title, and legend
    plt.title('Bland-Altman Plot')
    plt.xlabel('Mean of Two Measurements')
    plt.ylabel('Difference Between Measurements (Data 1 - Data 2)')
    plt.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plt.show()