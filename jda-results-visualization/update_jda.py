# 按任务顺序更新 ours.JDA 值
JDA_VALUES = [
    50.45, 66.73,  # Digit: USPS→MNIST, MNIST→USPS
    89.44, 88.06,  # COIL: COIL1→COIL2, COIL2→COIL1
    62.49, 59.19, 86.86, 50.01,  # PIE 1-4
    60.26, 64.09, 82.85, 39.95,  # PIE 5-8
    51.05, 59.24, 68.85, 33.64,  # PIE 9-12
    80.58, 82.87, 87.01, 52.88,  # PIE 13-16
    46.46, 43.59, 52.21, 56.59,  # PIE 17-20
    43.95, 37.63, 45.22, 39.36, 38.98, 39.49,  # SURF 1-6
    31.17, 32.78, 89.17, 31.61, 32.67, 89.49   # SURF 7-12
]

import json
with open('_embedded_data.txt', 'r', encoding='utf-8') as f:
    data = json.load(f)

idx = 0
for ds_name in ['Digit', 'COIL', 'PIE', 'SURF/Office']:
    tasks = data[ds_name]
    for task_name in tasks:
        if idx < len(JDA_VALUES):
            data[ds_name][task_name]['ours']['JDA'] = JDA_VALUES[idx]
            idx += 1

if idx != 36:
    print(f'WARNING: Updated {idx} tasks, expected 36')
else:
    print('Updated 36 JDA values')

with open('_embedded_data.txt', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False)
