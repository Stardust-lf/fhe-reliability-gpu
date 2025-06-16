import os
import re
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Gill Sans'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['font.size'] = 16

# Pattern to match top entries: capture absolute count and function name
pattern = re.compile(r'^\s*([\d,]+)\s+\([\d.]+%\)\s+(?:\S+?:)?([^\(]+)\(')

data = {}
all_funcs = set()

# Parse each .info file in the current directory
for fname in os.listdir('.'):
    if not fname.endswith('.info'):
        continue
    op_name = fname[:-5]  # strip .info
    data[op_name] = {}
    with open(fname, 'r') as f:
        count = 0
        for line in f:
            m = pattern.match(line)
            if m and count < 5:
                instr_count = int(m.group(1).replace(',', ''))
                full_func = m.group(2).strip()
                simple_func = full_func.split("::")[-1]
                data[op_name][simple_func] = instr_count
                all_funcs.add(simple_func)
                count += 1
            if count >= 5:
                break

# Create DataFrame: rows are operations, columns are functions
df = pd.DataFrame.from_dict(data, orient='index', columns=sorted(all_funcs)).fillna(0)

# Plot stacked bar chart of absolute instruction counts with tab10 colormap
fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
df.plot(kind='bar', stacked=True, colormap='tab10', ax=ax, rot=45)
ax.set_ylabel("Instruction Count")
ax.set_xlabel("Operation")
ax.grid(True)
plt.tight_layout()
plt.savefig("BGV_profile.png")
plt.show()
