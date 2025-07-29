import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.widgets import RadioButtons

# Updated data including dyadic_prod
stages = {
    'Dotprod':    {'Rotate': 62.72, 'HMul': 25.08, 'Relin': 5.52, 'ApplyGalois': 2.90, 'HAdd': 1.59, 'ModSwitch': 0.48, 'switch_key': 0,   'dyadic_prod': 0},
    'Rotate':     {'ntt': 31.66,    'intt': 29.78,    'switch_key': 19.77, 'mod_poly': 2.97, 'mul_poly': 2.55, 'add_poly': 2.40, 'apply_galois': 2.02, 'dyadic_prod': 0},
    'HMul':       {'ntt': 24.34,    'intt': 16.72,    'switch_key': 0,     'mul_poly': 3.91, 'add_poly': 2.73, 'dot_prod_mod': 21.23, 'fastbconv_sk': 2.53, 'fast_floor': 1.38, 'dyadic_prod': 10.43},
    'Relin':      {'ntt': 32.72,    'intt': 30.78,    'switch_key': 20.43, 'mod_poly': 3.07, 'mul_poly': 2.63, 'add_poly': 2.48, 'dyadic_prod': 0},
    'ApplyGalois':{'ntt': 31.66,    'intt': 29.79,    'switch_key': 19.77, 'mod_poly': 2.97, 'mul_poly': 2.58, 'add_poly': 2.40, 'apply_galois': 2.02, 'dyadic_prod': 0},
    'HAdd':       {'add_poly': 99.15, 'dyadic_prod': 0},
    'ModSwitch':  {'mod_poly': 15.12, 'mul_poly': 16.06, 'add_poly': 6.80, 'memcpy': 20.44, 'sub_poly': 26.46, 'memset': 13.86, 'switch_key': 0, 'dyadic_prod': 0}
}

# Split switch_key into apply_evalk and baseconv
apply_ratio, base_ratio = 0.541, 0.459
for d in stages.values():
    if 'switch_key' in d:
        orig = d.pop('switch_key')
        if orig > 0:
            d['apply_evalk'] = orig * apply_ratio
            d['baseconv'] = orig * base_ratio

# Compute 'Others' for each stage
for d in stages.values():
    total = sum(d.values())
    d['Others'] = max(0, 100 - total)

# Create a consistent color map
unique_labels = sorted({lbl for d in stages.values() for lbl in d})
cmap = plt.get_cmap('tab20')
label_colors = {lbl: cmap(i / len(unique_labels)) for i, lbl in enumerate(unique_labels)}

# Set up figure and subplots
fig, axes = plt.subplots(4, 2, figsize=(14, 16))
axes = axes.flatten()
legend_ax = axes[-1]
legend_ax.axis('off')

# Add RadioButtons for interactivity
rax = fig.add_axes([0.92, 0.3, 0.07, 0.4])
radio = RadioButtons(rax, ['None'] + unique_labels)

def update(highlight):
    # Redraw all donut charts
    for ax, (stage, data) in zip(axes[:-1], stages.items()):
        ax.clear()
        labels, sizes = zip(*data.items())
        colors = [label_colors[lbl] for lbl in labels]
        alphas = [1.0 if (lbl == highlight or highlight == 'None') else 0.3 for lbl in labels]
        wedges, _ = ax.pie(
            sizes, startangle=90, wedgeprops={'width':0.5}, colors=colors
        )
        for w, a in zip(wedges, alphas):
            w.set_alpha(a)
        ax.set_title(stage)
    # Update unified legend
    legend_ax.clear()
    legend_ax.axis('off')
    handles = [
        Patch(color=label_colors[lbl], label=lbl, alpha=1.0 if (lbl == highlight or highlight == 'None') else 0.3)
        for lbl in unique_labels
    ]
    legend_ax.legend(handles, unique_labels, ncol=2, loc='center', frameon=False)
    fig.canvas.draw_idle()

radio.on_clicked(update)
update('None')

plt.tight_layout(rect=[0, 0, 0.90, 1])
plt.show()
