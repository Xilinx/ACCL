import re
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# keywords = ["Broadcast", "Allreduce", "AlltoAll",  ]
# keywords = ["Broadcast", "Allreduce" ]
keywords = ["Allreduce", "AlltoAll",  ]
# parts = ["lib","barrier","total"]
parts = ["lib","barrier","copy","lock","init", "type", "total"]

part_pattern = re.compile(r"(.*)_tensor durationUs:.*")

measurement_pattern = re.compile(r".*_tensor durationUs:(.*)")

# keyword_pattern = re.compile(r"Starting (Broadcast|Allreduce|AlltoAll)")
keyword_pattern = re.compile(r"Starting (Allreduce|AlltoAll)")
# keyword_pattern = re.compile(r"Starting (Broadcast|Allreduce)")

log_file_path = './accl_log/rank_0_stderr'

with open(log_file_path, 'r') as log_file:
    lines = log_file.readlines()

current_keyword = None
# results = { "Broadcast": {}, "Allreduce": {}, "AlltoAll": {} }
# results = { "Broadcast": {}, "Allreduce": {}}
results = { "Allreduce": {}, "AlltoAll": {} }

# averages = { "Broadcast": {}, "Allreduce": {}}
# averages = { "Broadcast": {}, "Allreduce": {}, "AlltoAll": {} }
averages = { "Allreduce": {}, "AlltoAll": {} }

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
            if part=='total':
                current_keyword = None

for op, parts in results.items():
    for part, values in parts.items():
        test_sum = 0
        count = 0
        for el in values:
            test_sum +=  el
            count += 1
        averages[op][part] = test_sum / count    

for op, part in averages.items():
    labels = [key for key in part if key != 'total' and key != 'barrier']
    times = [part[key] for key in labels]
    total_time = part['total'] - part['barrier']
    other_time = total_time - sum(times)

    if other_time > 0:
        labels.append('Other')
        times.append(other_time)
    
    plt.figure()
    plt.pie(times, labels=labels, autopct=lambda p: f'{p * total_time / 100:.2f}us')
    plt.title(f'Runtime Distribution for {op}')

    plt.savefig('composition_' + op + '_plot.png')
