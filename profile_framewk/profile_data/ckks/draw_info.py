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

data = {}
all_funcs = set()

# Find all .info files
info_files = [f for f in os.listdir('.') if f.endswith('.info')]

for fname in info_files:
    with open(fname, 'r') as f:
        lines = f.readlines()

    # Locate the "file:function" header line
    header_idx = next((i for i, line in enumerate(lines) if 'file:function' in line), None)
    if header_idx is None:
        print(f"Warning: no 'file:function' header in {fname}, skipping")
        continue

    # Start parsing two lines below the header
    start_idx = header_idx + 2

    op_name = fname[:-5]  # strip ".info"
    data[op_name] = {}
    matches = 0

    # Iterate until next separator or we've collected 5 entries
    for line in lines[start_idx:]:
        if line.strip().startswith('---'):
            break
        m = item_re.match(line)
        if not m:
            continue

        # raw function token including qualifiers/args
        raw = m.group(2).strip()
        # strip off argument list if present
        raw = raw.split('(')[0]
        # extract only the final identifier
        if '::' in raw:
            simple = raw.split('::')[-1]
        elif ':' in raw:
            simple = raw.split(':')[-1]
        else:
            simple = raw

        instr_count = int(m.group(1).replace(',', ''))
        data[op_name][simple] = instr_count
        all_funcs.add(simple)
        matches += 1
        if matches >= 5:
            break

    if matches == 0:
        print(f"Warning: no entries found in {fname}")

# Build DataFrame: rows = operations, cols = function names
df = pd.DataFrame.from_dict(
    data,
    orient='index',
    columns=sorted(all_funcs)
).fillna(0)

# Sort operations (the index) alphabetically
df = df.sort_index()

# Plot stacked bar chart
fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
df.plot(
    kind='bar',
    stacked=True,
    colormap='tab10',
    ax=ax,
    rot=30,
    edgecolor='black'
)

# Right-align x-tick labels
plt.setp(ax.get_xticklabels(), ha='right')

ax.set_xlabel("Operation")
ax.set_ylabel("Instruction Count")
ax.set_title("CKKS")
ax.legend(loc='upper left')
ax.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig("CKKS_profile.png")
plt.show()
