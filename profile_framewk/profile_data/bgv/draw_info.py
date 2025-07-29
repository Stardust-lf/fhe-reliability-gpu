import os
import re
import pandas as pd
import matplotlib.pyplot as plt

# --- Plot styling ---
plt.rcParams['font.family'] = 'Gill Sans'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['font.size'] = 16

# Regex to capture "count", "%", and "file:function" up to the "[" bracket
item_re = re.compile(r'^\s*([\d,]+)\s+\([\d.]+%\)\s+(.+?)(?=\s+\[)')

# 1) Identify all .info files (operations)
info_files = [f for f in os.listdir('.') if f.endswith('.info')]

# 2) For each file, extract Top-5 functions from each of its sections, then compute the average across sections
top5_means = {}

for fname in info_files:
    lines = open(fname, 'r').read().splitlines()
    header_idxs = [i for i, line in enumerate(lines) if 'file:function' in line]
    section_dicts = []

    for hidx in header_idxs:
        counts = {}
        cnt = 0
        for L in lines[hidx+2:]:
            if 'file:function' in L or L.strip().startswith('---'):
                break
            m = item_re.match(L)
            if not m:
                continue
            raw = m.group(2).split('(')[0].strip()
            name = raw.split('::')[-1].split(':')[-1]
            instr = int(m.group(1).replace(',', ''))
            counts[name] = instr
            cnt += 1
            if cnt >= 5:
                break

        if not counts:
            print(f"Warning: no entries in section starting at line {hidx+1} of {fname}")
        section_dicts.append(counts)

    # Build a DataFrame for this file: rows = sections, columns = functions
    df_file = pd.DataFrame(section_dicts).fillna(0)
    # Compute mean instruction count per function
    mean_series = df_file.mean(axis=0)
    # Select top 5 functions by average
    top5 = mean_series.nlargest(5)
    top5_means[fname[:-5]] = top5

# 3) Construct final DataFrame: rows = operations, columns = functions
final_df = pd.DataFrame.from_dict(top5_means, orient='index').fillna(0)

# 4) Plot stacked bar chart: one bar per operation showing its average Top-5 breakdown
fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
final_df.plot(
    kind='bar',
    stacked=True,
    ax=ax,
    edgecolor='black'
)

plt.setp(ax.get_xticklabels(), ha='right')
ax.set_xlabel("Operation (file name)")
ax.set_ylabel("Average Instruction Count of Top-5 Functions")
ax.set_title("Per-Operation Average Top-5 Function Instruction Counts")
ax.legend(title="Function", bbox_to_anchor=(1.05, 1), loc='upper left')
ax.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
# plt.savefig("BGV_top5_per_operation.png")
plt.show()
