import os
import re
import pandas as pd
import matplotlib.pyplot as plt

# --- Plot styling ---
plt.rcParams['font.family'] = 'Gill Sans'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['font.size'] = 12

item_re = re.compile(r'^\s*([\d,]+)\s+\([\d.]+%\)\s+(.+?)(?=\s+\[)')
profile_sets = {'BFV': 'bfv', 'BGV': 'bgv', 'CKKS': 'ckks'}

# First pass: collect final DataFrames for each profile
profile_dfs = {}
for label, folder in profile_sets.items():
    dir_path = os.path.join(os.getcwd(), folder)
    if not os.path.isdir(dir_path):
        continue
    info_files = [f for f in os.listdir(dir_path) if f.endswith('.info')]
    top5_means = {}
    for fname in info_files:
        lines = open(os.path.join(dir_path, fname)).read().splitlines()
        header_idxs = [i for i, L in enumerate(lines) if 'file:function' in L]
        section_dicts = []
        for hidx in header_idxs:
            counts = {}
            cnt = 0
            for L in lines[hidx+2:]:
                if 'file:function' in L or L.strip().startswith('---'):
                    break
                m = item_re.match(L)
                if not m: continue
                raw = m.group(2).split('(')[0].strip()
                name = raw.split('::')[-1].split(':')[-1]
                counts[name] = int(m.group(1).replace(',', ''))
                cnt += 1
                if cnt >= 5:
                    break
            if counts:
                section_dicts.append(counts)
        df_file = pd.DataFrame(section_dicts).fillna(0)
        top5 = df_file.mean().nlargest(5)
        top5_means[fname[:-5]] = top5
    final_df = pd.DataFrame.from_dict(top5_means, orient='index').fillna(0)
    profile_dfs[label] = final_df

# Compute common operations across all profiles
sets = [set(df.index) for df in profile_dfs.values()]
common_ops = sorted(set.intersection(*sets))

# Create 1x3 subplots
fig, axes = plt.subplots(1, 3, figsize=(12, 8), dpi=150, sharey=True)

for ax, (label, df) in zip(axes, profile_dfs.items()):
    # Determine ordering: common ops first, then the rest
    rest = [op for op in df.index if op not in common_ops]
    ordered = common_ops + rest
    df_ord = df.reindex(ordered).fillna(0)

    df_ord.plot(kind='bar', stacked=True, edgecolor='black',
                ax=ax, legend=False, colormap='tab20c')
    ax.set_title(label)
    ax.set_xlabel("")
    ax.grid(alpha=0.6)
    ax.set_xticklabels(df_ord.index, rotation=45, ha='right', fontsize=10)

axes[0].set_ylabel("Avg Instruction Count")

# Place legend vertically on right
# handles, labels = axes[-1].get_legend_handles_labels()
all_handles = []
all_labels = []
for ax in axes:
    h, l = ax.get_legend_handles_labels()
    for hi, li in zip(h, l):
        if li not in all_labels:
            all_handles.append(hi)
            all_labels.append(li)

fig.legend(all_handles, all_labels, title="Function",
           loc='center right', bbox_to_anchor=(1, 0.5), frameon=False)
# fig.legend(handles, labels, title="Function",
#            loc='center right', bbox_to_anchor=(1, 0.5),
#            frameon=False, ncol=1)

# Adjust margins to fit legend
fig.subplots_adjust(left=0.05, right=0.65, bottom=0.20, top=0.90)
plt.savefig("full_profile.png")
plt.show()
