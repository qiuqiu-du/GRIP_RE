import numpy as np
import glob
import os
def calculate_heading(input_file, output_file):
    # 读取数据（frame_id, object_id, object_type, x, y）
    data = np.loadtxt(input_file, delimiter=" ")

    # 获取所有唯一的object_id
    object_ids = np.unique(data[:, 1])

    # 初始化输出数据（10列）
    output_data = np.zeros((data.shape[0], 10))
    output_data[:, :5] = data  # 前5列保持不变

    # 遍历每个object_id，计算heading
    for obj_id in object_ids:
        # 提取当前object的所有轨迹点
        obj_data = data[data[:, 1] == obj_id]
        frames = obj_data[:, 0].astype(int)

        # 遍历每一帧，计算heading
        for i in range(len(frames)):
            current_frame = frames[i]
            current_x, current_y = obj_data[i, 3], obj_data[i, 4]

            # 尝试获取下一帧数据
            if i + 1 < len(frames):
                next_x, next_y = obj_data[i + 1, 3], obj_data[i + 1, 4]
                delta_x = next_x - current_x
                delta_y = next_y - current_y
            # 如果没有下一帧，尝试获取前一帧数据
            elif i - 1 >= 0:
                prev_x, prev_y = obj_data[i - 1, 3], obj_data[i - 1, 4]
                delta_x = current_x - prev_x
                delta_y = current_y - prev_y
            # 如果只有一帧数据，heading=0
            else:
                delta_x, delta_y = 0.001, 0  # 避免除以0

            # 计算heading（弧度）
            if delta_x == 0:
                delta_x = 0.001  # 避免除以0
            heading = np.arctan2(delta_y, delta_x)

            # 找到当前帧在原始数据中的位置，并写入heading
            idx = np.where((data[:, 1] == obj_id) & (data[:, 0] == current_frame))[0][0]
            output_data[idx, 9] = heading  # 第10列存储heading

    # 保存结果（第6-9列保持0）
    np.savetxt(
        output_file,
        output_data,
        delimiter=" ",
        fmt="%d %d %d %.6f %.6f %.1f %.1f %.1f %.1f %.6f"
    )

def filter_sort_and_reindex(input_file, output_file, keep_values):
    filtered_lines = []

    with open(input_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if parts and int(parts[0]) in keep_values:
                filtered_lines.append(parts)

    # 按原来的第一列值排序
    filtered_lines.sort(key=lambda x: int(x[0]))

    # 重新编号第一列
    new_index = 1
    prev_value = None
    mapping = {}

    for parts in filtered_lines:
        original_value = int(parts[0])
        if original_value != prev_value:
            mapping[original_value] = new_index
            prev_value = original_value
            new_index += 1
        parts[0] = str(mapping[original_value])

    with open(output_file, 'w') as f_out:
        for parts in filtered_lines:
            f_out.write(" ".join(parts) + "\n")

# 指定需要保留的第一列的值
# gt result
keep_values = {1, 3, 6, 8, 11, 13,}
#hist
# keep_values = {3, 6, 8, 11, 13, 16}

# 运行脚本
input_file = "lane/result_v.txt"  # 替换成你的输入文件名
# output_file = "lane/gt_v.txt"  # 输出文件名

# input_file = sorted(glob.glob(os.path.join('lane', '*.txt')))

filter_sort_and_reindex(input_file, input_file, keep_values)
calculate_heading(input_file, input_file)

