import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import sys


def choose_shape(traj_dot):
    '''
    传入数据行的第三列必须是类别
    '''

    object_type = traj_dot[0][2]
    # print(traj_dot[2])
    if object_type == 1 or object_type == 2:
        dot_shape = 'o'
    elif object_type == 3:
        dot_shape = '*'
    elif object_type == 4:
        dot_shape = 'd'
    elif object_type == 5:
        dot_shape = 's'
    else:
        dot_shape = 'x'
    return dot_shape

# 读取轨迹数据的函数
def load_data(filename):
    data = np.loadtxt(filename)
    return data

# 筛选最后一帧出现的物体编号
def get_last_frame_objects(history_data):
    last_frame = int(np.max(history_data[:, 0]))  # 获取最后一帧的编号
    last_frame_data = history_data[history_data[:, 0] == last_frame]
    objects_in_last_frame = np.unique(last_frame_data[:, 1])  # 获取最后一帧中所有物体的编号
    filtered_objects = objects_in_last_frame[~np.isin(objects_in_last_frame, object_filter)]
    return filtered_objects

# 根据物体编号筛选轨迹数据,返回字典
def filter_data_by_objects(data, object_ids):
    filtered_data = {}
    for obj_id in object_ids:
        filtered_data[obj_id] = data[data[:, 1] == obj_id, :5]  # 保留前五列
    return filtered_data

# 零中心化数据，使用历史轨迹的均值
def zero_center_data_by_history(history_data, data):
    # 计算历史轨迹的均值
    history_mean = np.mean(history_data[:, 3:5], axis=0)  # 只对X和Y坐标进行零中心化
    # 将数据减去历史轨迹的均值
    data[:, 3:5] -= history_mean
    return data

# 绘制轨迹
def plot_trajectory(history_data, gt_data, result_data):
    # 获取最后一帧的物体编号
    last_frame_objects = get_last_frame_objects(history_data)
    # print(*last_frame_objects)

    # 筛选出这些物体的轨迹
    history_filtered = filter_data_by_objects(history_data, last_frame_objects)
    # print(history_filtered)
    gt_filtered = filter_data_by_objects(gt_data, last_frame_objects)
    result_filtered = filter_data_by_objects(result_data, last_frame_objects)

    # 对历史轨迹、GT和预测轨迹进行零中心化，以历史轨迹的均值为参考点
    for obj_id in history_filtered:
        history_filtered[obj_id] = zero_center_data_by_history(history_data, history_filtered[obj_id])
    for obj_id in gt_filtered:
        gt_filtered[obj_id] = zero_center_data_by_history(history_data, gt_filtered[obj_id])
    for obj_id in result_filtered:
        result_filtered[obj_id] = zero_center_data_by_history(history_data, result_filtered[obj_id])

    # 绘图
    plt.figure(figsize=config.fig_size, dpi=120)
    plt.xlabel('X Coordinate', fontsize=config.font_size)  # 横坐标标题
    plt.ylabel('Y Coordinate', fontsize=config.font_size)  # 纵坐标标题
    plt.xticks(fontsize=config.font_size)  # 增大横坐标刻度字号
    plt.yticks(fontsize=config.font_size)  # 增大纵坐标刻度字号

    # 绘制每个物体的历史轨迹
    for obj_id in history_filtered:
        obj_history = history_filtered[obj_id]
        dot_shape = choose_shape(obj_history)

        plt.plot(obj_history[:, 3], obj_history[:, 4], color=config.history_color, linewidth=config.line_width, marker=dot_shape, markersize=config.dot_size,
                 markerfacecolor='gray', zorder=6)

        # 获取历史轨迹的最后一点
        last_point = obj_history[-1, 3:5]

        # 加粗最后一个点
        # plt.scatter(last_point[0], last_point[1], color=config.history_color, s=15, marker=config.dot_shape, zorder=5)  # s=100是点的大小，zorder=5确保它在其他点的前面

            # 获取对应物体的GT轨迹的起始点
        if obj_id in gt_filtered:
            try:
                obj_gt = gt_filtered[obj_id]
                first_gt_point = obj_gt[0, 3:5]  # GT的起始点
                plt.plot([last_point[0], first_gt_point[0]], [last_point[1], first_gt_point[1]], color=config.gt_color, linewidth=config.line_width, zorder=5, alpha=0.85)  # 连接线
                if config.obj_label:
                    # 计算标签位置（线段中点）
                    label_x = (last_point[0] + first_gt_point[0]) / 2
                    label_y = (last_point[1] + first_gt_point[1]) / 2

                    # 添加标签
                    plt.text(label_x, label_y,
                             f'{obj_id:.0f}',
                             color=config.gt_color,
                             fontsize=6,
                             ha='center',
                             va='center',
                             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=2),
                             zorder=10)
            except:
                print(f'Can not find first GT of {obj_id:.0f}')
                continue






        # 获取对应物体的Result轨迹的起始点
        if obj_id in result_filtered:
            try:
                obj_result = result_filtered[obj_id]
                first_result_point = obj_result[0, 3:5]  # Result的起始点
                plt.plot([last_point[0], first_result_point[0]], [last_point[1], first_result_point[1]], color=config.result_color, linewidth=config.line_width, zorder=4)  # 连接线
            except:
                print(f'Can not find first RESULT of {obj_id:.0f}')
                continue

    # 绘制每个物体的预测轨迹
    for obj_id in result_filtered:
        obj_result = result_filtered[obj_id]
        dot_shape = choose_shape(obj_result)
        plt.plot(obj_result[:, 3], obj_result[:, 4], color=config.result_color, linewidth=config.line_width,  marker=dot_shape, markersize=config.dot_size,
                 markerfacecolor='green', zorder=4)  # 细线，绿色为预测轨迹

    # 绘制每个物体的Ground Truth轨迹
    for obj_id in gt_filtered:
        obj_gt = gt_filtered[obj_id]
        dot_shape = choose_shape(obj_gt)
        plt.plot(obj_gt[:, 3], obj_gt[:, 4], color=config.gt_color, linewidth=config.line_width, marker=dot_shape, markersize=config.dot_size,
                 markerfacecolor='red', zorder=5, alpha=0.85)    # 细线，红色为GT轨迹


    # 创建一个自定义的图例项，仅包含一个点，标注为 'bike'
    legend_point_ped = Line2D([0], [0], marker='*', color='green', markerfacecolor='green', markersize=config.dot_size,linewidth=config.line_width,
                          label='Pedestrian')
    legend_point_bik = Line2D([0], [0], marker='d', color='gray', markerfacecolor='gray', markersize=config.dot_size,linewidth=config.line_width,
                          label='Bike')
    legend_point_veh = Line2D([0], [0], marker='o', color='brown', markerfacecolor='brown', markersize=config.dot_size,linewidth=config.line_width,
                          label='Vehicle')
    legend_point_oth = Line2D([0], [0], marker='s', color='red', markerfacecolor='red', markersize=config.dot_size,linewidth=config.line_width,
                          label='Others')

    # 设置坐标轴刻度和网格
    plt.grid(True)


    # 设置标题、标签
    # plt.title('Vehicle Trajectory Prediction')
    # plt.xlabel('X Position')
    # plt.ylabel('Y Position')

    # 添加图例
    history_line = plt.Line2D([0], [0], color='gray', lw=2, label='History')
    gt_line = plt.Line2D([0], [0], color='red', lw=2, label='Ground Truth')
    result_line = plt.Line2D([0], [0], color='green', lw=2, label='Prediction')

    plt.legend(handles=[ legend_point_ped], fontsize=config.font_size,) #history_line, gt_line, result_line,legend_point ,history_line, gt_line, result_line,legend_point_bik
    plt.tight_layout()
    plt.show()





# 数据文件路径
history_filename = './history.txt'
gt_filename = './gt.txt'
result_filename = './result.txt'

config = {
    'line_width': 4, # 3
    'history_color': 'gray',
    'gt_color': 'red',
    'result_color': 'green',
    'dot_shape': '*',
    'dot_size': 10,  # 6.5
    'fig_size':(10,7.5),#4.5
    'obj_label':False,
    'font_size': 28
}
config = type('Config', (object,), config)()

object_filter = [171092]

# 读取数据
history_data = load_data(history_filename)
gt_data = load_data(gt_filename)
result_data = load_data(result_filename)

# 绘制轨迹
plot_trajectory(history_data, gt_data, result_data)
