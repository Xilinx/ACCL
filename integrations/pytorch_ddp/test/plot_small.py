import re
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.ticker as mticker

keywords = ["Broadcast", "Allreduce" ]
# parts = ["lib","barrier","total"]
parts = ["lib", "copy","init", "total", "device", "pytorch", "sleep"]
parts_plot = ["init", "device", "lib_oh", "copy", "total_oh", "pytorch_oh"]


part_pattern = re.compile(r"(.*)_.*_.* durationUs: .*")
op_pattern = re.compile(r".*_(.*)_.* durationUs: .*")
count_pattern = re.compile(r".*_.*_(.*) durationUs: .*")

measurement_pattern = re.compile(r".*_.*_.* durationUs: (.*)")


log_file_path = './accl_log/rank_0_stderr'

with open(log_file_path, 'r') as log_file:
    lines = log_file.readlines()

current_keyword = None
results = { "Broadcast": {}, "Allreduce": {}}
averages = { "Broadcast": {}, "Allreduce": {}}

# results = { "Broadcast": {}, "Allreduce": {}}

sizes = []

for op in results:
    for part in parts:
        results[op][part] = {}
        averages[op][part] = {}
    for part in parts_plot:        
        averages[op][part] = {}
for line in lines:
    part_match = part_pattern.search(line)
    op_match = op_pattern.search(line)
    count_match = count_pattern.search(line)
    if (not part_match) or (not op_match) or (not count_match):
        continue
    part = part_match.group(1).strip()
    op = op_match.group(1).strip()
    cnt = int(count_match.group(1).strip())
    if cnt > 2097152:
        continue
    if op not in keywords:
        continue
    
    if cnt not in sizes:
        sizes.append(cnt)
    if part in parts:
        measurement_match = measurement_pattern.search(line)
        measurement = measurement_match.group(1).strip()
        if cnt not in results[op][part].keys():
            results[op][part][cnt] = []
        results[op][part][cnt].append(float(measurement))

        
for op, parts in results.items():
    for part, cnts in parts.items():
        for cnt, mes in cnts.items():
            test_sum = 0
            count = 0
            for el in mes:
                test_sum +=  el
                count += 1
            averages[op][part][cnt] = test_sum / count    

for op, parts in averages.items():
    for cnt in sizes:
        averages[op]['lib_oh'][cnt] = parts['lib'][cnt] - parts['device'][cnt]
        averages[op]['total_oh'][cnt] = parts['total'][cnt] - parts['sleep'][cnt]  - parts['lib'][cnt] - parts['init'][cnt] - parts['copy'][cnt]
        averages[op]['pytorch_oh'][cnt] = parts['pytorch'][cnt] - (parts['total'][cnt])

    averages[op].pop('lib')
    averages[op].pop('total')
    averages[op].pop('sleep')
    averages[op].pop('pytorch')    


sizes.sort()

av_lists = {}
for word in keywords:
    av_lists[word] = {}
    for part in parts_plot:
        av_lists[word][part] = []
        for size in sizes:
            av_lists[word][part].append(averages[word][part][size])



# print(av_lists['Allreduce'])
# print(av_lists['Allreduce'].values().shape)
# print(sizes)
# print(sizes.shape)

for op in keywords:
    fig, ax = plt.subplots()
    ax.stackplot(sizes, av_lists[op].values(),
                 labels=av_lists[op].keys(), alpha=0.8)
    ax.legend(loc='upper left', reverse=True)
    plt.gca().set_xscale('log', base=2)
    ax.set_title('Execution time composition' + )
    ax.set_xlabel('size[B]')
    ax.set_ylabel('Latency us')
    # add tick at every 200 million people
    # ax.yaxis.set_minor_locator(mticker.MultipleLocator(.2))

    plt.savefig(op + '_composition.png')

# for i, (dict_name, sub_dict) in enumerate(results.items()):
    # for j, (key, values) in enumerate(sub_dict.items()):
        # sns.histplot(values, ax=axes[i, j], bins=20, stat='percent', kde=True)
        # axes[i, j].set_title(f'{dict_name} - {key}')

# plt.tight_layout()
# plt.savefig("fullplot.png")

            
