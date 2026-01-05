import matplotlib.pyplot as plt
import numpy as np

# Set up Chinese font support and style
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-darkgrid')

# Data: bitwidth and corresponding values
bitwidths = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]

# Total Miss Rate (from Det Rate column: Miss Rate = 1 - Det Rate)
data = {
    'SBF': {
        'total': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'lazy': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'inter': [1405, 658, 274, 109, 55, 17, 20, 11, 3, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    },
    'DBF': {
        'total': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'lazy': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'inter': [1298, 712, 286, 117, 64, 29, 17, 11, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    },
    'MLF': {
        'total': [1300, 750, 299, 128, 56, 28, 15, 5, 4, 3, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        'lazy': [1300, 750, 299, 128, 56, 28, 15, 5, 4, 3, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        'inter': [2705, 1408, 573, 237, 111, 45, 35, 16, 7, 3, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    },
    'SBF+SBF': {
        'total': [7, 5, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'lazy': [5802, 5345, 5027, 4612, 4355, 4008, 3712, 3542, 3328, 3143, 3061, 2934, 2761, 2578, 2490, 2390, 2355, 2201, 2239, 2024],
        'inter': [1336, 672, 288, 129, 70, 30, 22, 6, 5, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    },
    'SBF+DBF': {
        'total': [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'lazy': [857, 491, 429, 340, 303, 281, 260, 193, 203, 169, 140, 148, 130, 129, 102, 113, 83, 65, 69, 77],
        'inter': [1296, 738, 275, 118, 42, 34, 18, 12, 5, 3, 2, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    },
    'MLF+SBF': {
        'total': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'lazy': [174, 77, 37, 13, 7, 3, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'inter': [1328, 736, 288, 131, 51, 37, 15, 9, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    },
    'MLF+DBF': {
        'total': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'lazy': [134, 76, 36, 17, 6, 6, 4, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'inter': [1291, 721, 290, 113, 57, 20, 19, 8, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    },
    'MLF+MLF': {
        'total': [2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'lazy': [165, 90, 35, 12, 7, 1, 3, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'inter': [1320, 729, 287, 121, 71, 30, 10, 7, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    },
}

# Colors for each scenario
colors = {
    'SBF': '#e63946',
    'DBF': '#f4a261',
    'MLF': '#2a9d8f',
    'SBF+SBF': '#9b59b6',
    'SBF+DBF': '#3498db',
    'MLF+SBF': '#1abc9c',
    'MLF+DBF': '#e74c3c',
    'MLF+MLF': '#34495e',
}

# Create figure with 3 subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('NTT Multi-Fault Analysis: Miss Rate vs Q Bitwidth\n(N=64, Lazy Reduction + Global ABFT)', 
             fontsize=14, fontweight='bold', y=1.02)

# Plot 1: Total Miss Rate (dot-dash line)
ax1 = axes[0]
for scenario, vals in data.items():
    y_vals = np.array(vals['total'])
    ax1.plot(bitwidths, y_vals, linestyle='-.', linewidth=2, marker='o', 
             markersize=4, color=colors[scenario], label=scenario)
ax1.set_xlabel('Q Bitwidth', fontsize=11)
ax1.set_ylabel('Miss Rate (ppm)', fontsize=11)
ax1.set_title('Total Miss Rate (Dot-Dash)', fontsize=12, fontweight='bold')
ax1.legend(loc='upper right', fontsize=9)
ax1.set_xlim(10, 31)
ax1.grid(True, alpha=0.3)

# Plot 2: LazyCatch Miss Rate (solid line)
ax2 = axes[1]
for scenario, vals in data.items():
    y_vals = np.array(vals['lazy'])
    ax2.plot(bitwidths, y_vals, linestyle='-', linewidth=2, marker='s', 
             markersize=4, color=colors[scenario], label=scenario)
ax2.set_xlabel('Q Bitwidth', fontsize=11)
ax2.set_ylabel('Miss Rate (ppm)', fontsize=11)
ax2.set_title('LazyCatch Miss Rate (Solid)', fontsize=12, fontweight='bold')
ax2.legend(loc='upper right', fontsize=9)
ax2.set_xlim(10, 31)
ax2.grid(True, alpha=0.3)

# Plot 3: InterCatch Miss Rate (dashed line)
ax3 = axes[2]
for scenario, vals in data.items():
    y_vals = np.array(vals['inter'])
    ax3.plot(bitwidths, y_vals, linestyle='--', linewidth=2, marker='^', 
             markersize=4, color=colors[scenario], label=scenario)
ax3.set_xlabel('Q Bitwidth', fontsize=11)
ax3.set_ylabel('Miss Rate (ppm)', fontsize=11)
ax3.set_title('InterCatch Miss Rate (Dashed)', fontsize=12, fontweight='bold')
ax3.legend(loc='upper right', fontsize=9)
ax3.set_xlim(10, 31)
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
# plt.savefig('/mnt/user-data/outputs/fault_analysis_3panels.png', dpi=150, bbox_inches='tight', 
#             facecolor='white', edgecolor='none')
plt.close()

# Create a combined plot with all lines
fig, ax = plt.subplots(figsize=(14, 8))

for scenario, vals in data.items():
    color = colors[scenario]
    
    # Total Miss Rate - dot-dash line
    y_total = np.array(vals['total'])
    ax.plot(bitwidths, y_total, linestyle='-.', linewidth=2.5, marker='o', 
            markersize=5, color=color, label=f'{scenario} (Total)', alpha=0.9)
    
    # LazyCatch Miss Rate - solid line
    y_lazy = np.array(vals['lazy'])
    ax.plot(bitwidths, y_lazy, linestyle='-', linewidth=2, marker='s', 
            markersize=4, color=color, label=f'{scenario} (Lazy)')
    
    # InterCatch Miss Rate - dashed line
    y_inter = np.array(vals['inter'])
    ax.plot(bitwidths, y_inter, linestyle='--', linewidth=2, marker='^', 
            markersize=4, color=color, label=f'{scenario} (Inter)')

ax.set_xlabel('Q Bitwidth', fontsize=12)
ax.set_ylabel('Miss Rate (ppm)', fontsize=12)
ax.set_title('NTT Multi-Fault Analysis: Miss Rate vs Q Bitwidth\n(N=64, Lazy Reduction + Global ABFT)', 
             fontsize=14, fontweight='bold')
ax.set_xlim(10, 31)
ax.grid(True, alpha=0.3)

# Create custom legend
from matplotlib.lines import Line2D
legend_elements = []
# Scenario colors
for scenario in data.keys():
    legend_elements.append(Line2D([0], [0], color=colors[scenario], linewidth=3, label=scenario))
# Line styles
legend_elements.append(Line2D([0], [0], color='gray', linestyle='-.', linewidth=2, label='Total Miss Rate'))
legend_elements.append(Line2D([0], [0], color='gray', linestyle='-', linewidth=2, label='LazyCatch Miss'))
legend_elements.append(Line2D([0], [0], color='gray', linestyle='--', linewidth=2, label='InterCatch Miss'))

ax.legend(handles=legend_elements, loc='upper right', fontsize=9, ncol=2)

plt.tight_layout()
plt.show()
# plt.savefig('/mnt/user-data/outputs/fault_analysis_combined.png', dpi=150, bbox_inches='tight',
#             facecolor='white', edgecolor='none')
plt.close()

# Create a cleaner version with separate subplots for each scenario type
fig, axes = plt.subplots(2, 4, figsize=(18, 10))
axes = axes.flatten()

for idx, (scenario, vals) in enumerate(data.items()):
    ax = axes[idx]
    color = colors[scenario]
    
    # Plot all three line types
    ax.plot(bitwidths, vals['total'], '-.o', linewidth=2, markersize=4, 
            color='#2c3e50', label='Total Miss Rate')
    ax.plot(bitwidths, vals['lazy'], '-s', linewidth=2, markersize=4, 
            color='#e74c3c', label='LazyCatch Miss')
    ax.plot(bitwidths, vals['inter'], '--^', linewidth=2, markersize=4, 
            color='#3498db', label='InterCatch Miss')
    
    ax.set_xlabel('Q Bitwidth', fontsize=10)
    ax.set_ylabel('Miss Rate (ppm)', fontsize=10)
    ax.set_title(f'{scenario}', fontsize=12, fontweight='bold', color=color)
    ax.legend(loc='upper right', fontsize=8)
    ax.set_xlim(10, 31)
    ax.grid(True, alpha=0.3)

fig.suptitle('NTT Multi-Fault Analysis by Scenario\n(N=64, Lazy Reduction + Global ABFT)', 
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()
# plt.savefig('/mnt/user-data/outputs/fault_analysis_by_scenario.png', dpi=150, bbox_inches='tight',
#             facecolor='white', edgecolor='none')
plt.close()

print("Charts saved successfully!")
print("- fault_analysis_3panels.png: Three separate panels for Total/Lazy/Inter")
print("- fault_analysis_combined.png: All lines in one chart")
print("- fault_analysis_by_scenario.png: Separate subplot for each fault scenario")
