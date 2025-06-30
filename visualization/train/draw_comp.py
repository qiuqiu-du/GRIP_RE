import matplotlib.pyplot as plt
import json
import numpy as np
from collections import deque
from matplotlib import rcParams

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']  # 使用黑体或Arial Unicode MS
rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def moving_average(data, window_size=5):
    """计算滑动平均值"""
    window = deque(maxlen=window_size)
    smoothed = []
    for value in data:
        window.append(value)
        smoothed.append(np.mean(window))
    return smoothed

with open('val_car.json', 'r') as f:
    data_pre = json.load(f)

with open('val_car_re.json', 'r') as g:
    data_re = json.load(g)

# 提取x和y坐标
x_pre = np.array([item[1] for item in data_pre])
y_pre = np.array([round(item[2], 4) for item in data_pre])


x_re = np.array([item[1] for item in data_re])
y_re = np.array([round(item[2], 4) for item in data_re])

# 计算滑动平均（窗口大小为5）
window_size = 8
y_pre_smoothed = moving_average(y_pre, window_size)
y_re_smoothed = moving_average(y_re, window_size)

# 创建图形
plt.figure(figsize=(12, 6))

# 绘制原始曲线
plt.plot(x_pre, y_pre,
         marker='o',
         linestyle='-',
         linewidth=1.5,
         markersize=4,
         color='#1f77b4',
         alpha=0.3,  # 半透明显示
         )
#label='Original Data'

plt.plot(x_re, y_re,
         marker='o',
         linestyle='-',
         linewidth=1.5,
         markersize=4,
         color='#1a5276',
         alpha=0.3,  # 半透明显示
         )

# 绘制平滑曲线
plt.plot(x_pre, y_pre_smoothed,
         linestyle='-',
         linewidth=2.5,
         color='#f2c55c',
         label='Previous')

plt.plot(x_re, y_re_smoothed,
         linestyle='-',
         linewidth=2.5,
         color='#b56060',
         label='Refined')

plt.xlabel('Epoch', fontsize=18)
plt.ylabel('Validation Loss', fontsize=18)
plt.title('Validation Loss for Vehicles', fontsize=18, pad=10)

# 设置x轴刻度间隔为10
max_x = int(np.ceil(max(x_re)/10.0)) * 10
plt.xticks(np.arange(0, max_x + 1, 10))

# 添加网格和图例
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(fontsize=16)

# 自动调整布局
plt.tight_layout()

# 保存和显示
plt.savefig('smoothed_line_chart.png', dpi=300, bbox_inches='tight')
plt.show()