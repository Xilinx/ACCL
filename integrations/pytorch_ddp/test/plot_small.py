import re
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


keywords = ["Broadcast", "Allreduce", "AlltoAll" ]
# keywords = ["Broadcast", "Allreduce" ]
# parts = ["lib","barrier","total"]
parts = ["lib","barrier","copy","lock","init", "type", "total"]

part_pattern = re.compile(r"(.*)_tensor durationUs:.*")

measurement_pattern = re.compile(r".*_tensor durationUs:(.*)")

keyword_pattern = re.compile(r"Starting (Broadcast|Allreduce|AlltoAll)")
# keyword_pattern = re.compile(r"Starting (Broadcast|Allreduce)")

log_file_path = './accl_log/rank_0_stderr'

with open(log_file_path, 'r') as log_file:
    lines = log_file.readlines()

current_keyword = None
results = { "Broadcast": {}, "Allreduce": {}, "AlltoAll": {} }
# results = { "Broadcast": {}, "Allreduce": {}}

for op in results:
    for part in parts:
        results[op][part] = []

for line in lines:
    keyword_match = keyword_pattern.search(line)
    if keyword_match:
        current_keyword = keyword_match.group(1)
        continue

    if current_keyword:
        part_match = part_pattern.search(line)
        if not part_match:
            continue
        part = part_match.group(1).strip()
        if part in parts:
            measurement_match = measurement_pattern.search(line)
            measurement = measurement_match.group(1).strip()
            results[current_keyword][part].append(float(measurement))

fig, axes = plt.subplots(len(keywords), len(parts), figsize=(5 * len(parts), 5 * len(keywords) ))

for i, (dict_name, sub_dict) in enumerate(results.items()):
    for j, (key, values) in enumerate(sub_dict.items()):
        sns.histplot(values, ax=axes[i, j], bins=20, stat='percent', kde=True)
        axes[i, j].set_title(f'{dict_name} - {key}')

plt.tight_layout()
plt.savefig("fullplot.png")

            
