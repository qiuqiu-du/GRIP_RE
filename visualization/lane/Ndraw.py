import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import matplotlib.patches as patches


def draw_rect(last_point, rect_width = 1.8, rect_height = 8.4, facecolor = 'black'):
    rect = patches.Rectangle(
        (last_point[0] - rect_width / 2, last_point[1] - rect_height / 2),
        rect_width, rect_height,
        linewidth=1, edgecolor=None, facecolor=facecolor, alpha=0.7, zorder=6
    )
    plt.gca().add_patch(rect)

def choose_shape(traj_dot):
    '''
    传入数据行的第三列必须是类别
    '''

    object_type = traj_dot[0][2]
    # print(traj_dot[2])
    if object_type == 1 or object_type == 2:
        dot_shape = None
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
    return objects_in_last_frame

# 根据物体编号筛选轨迹数据,返回字典
def filter_data_by_objects(data, object_ids):
    filtered_data = {}
    for obj_id in object_ids:
        obj_data = data[data[:, 1] == obj_id, :5]  # 筛选前五列
        if obj_data.shape[0] > 0:  # 仅在数据非空时添加
            filtered_data[obj_id] = obj_data
    return filtered_data


# 零中心化数据，使用历史轨迹的均值
def zero_center_data_by_history(history_data, data):
    # 计算历史轨迹的均值
    history_mean = np.mean(history_data[:, 3:5], axis=0)  # 只对X和Y坐标进行零中心化
    # 将数据减去历史轨迹的均值
    data[:, 3:5] -= history_mean
    return data

# 绘制轨迹
def plot_trajectory(history_data, gt_data, result_data1, result_data2=None, plot_second_result=False):
    # 获取最后一帧的物体编号
    last_frame_objects = get_last_frame_objects(history_data)
    # print(*last_frame_objects)

    # 筛选出这些物体的轨迹
    history_filtered = filter_data_by_objects(history_data, last_frame_objects)
    # print(history_filtered)
    gt_filtered = filter_data_by_objects(gt_data, last_frame_objects)
    result_filtered1 = filter_data_by_objects(result_data1, last_frame_objects)
    if plot_second_result:
        result_filtered2 = filter_data_by_objects(result_data2, last_frame_objects)

    # 对历史轨迹、GT和预测轨迹进行零中心化，以历史轨迹的均值为参考点
    for obj_id in history_filtered:
        history_filtered[obj_id] = zero_center_data_by_history(history_data, history_filtered[obj_id])
    for obj_id in gt_filtered:
        gt_filtered[obj_id] = zero_center_data_by_history(history_data, gt_filtered[obj_id])
    for obj_id in result_filtered1:
        result_filtered1[obj_id] = zero_center_data_by_history(history_data, result_filtered1[obj_id])
    if plot_second_result:
        for obj_id in result_filtered2:
            result_filtered2[obj_id] = zero_center_data_by_history(history_data, result_filtered2[obj_id])

    # 绘图
    plt.figure(figsize=config.fig_size, dpi=120)


    # 绘制每个物体的历史轨迹
    for obj_id in history_filtered:
        obj_history = history_filtered[obj_id]
        dot_shape = choose_shape(obj_history)

        plt.plot(obj_history[:, 3], obj_history[:, 4], color=config.history_color, linewidth=config.line_width, marker=dot_shape, markersize=config.dot_size,
                 markerfacecolor='black', zorder=5, alpha=0.7)

        # 获取历史轨迹的最后一点
        last_point = obj_history[-1, 3:5]

        draw_rect(last_point)

        # 加粗最后一个点
        # plt.scatter(last_point[0], last_point[1], color=config.history_color, s=15, marker=config.dot_shape, zorder=5)  # s=100是点的大小，zorder=5确保它在其他点的前面

        # 获取对应物体的GT轨迹的起始点
        if obj_id in gt_filtered:
            obj_gt = gt_filtered[obj_id]
            first_gt_point = obj_gt[0, 3:5]  # GT的起始点
            draw_rect(last_point, facecolor='blue')
            plt.plot([last_point[0], first_gt_point[0]], [last_point[1], first_gt_point[1]], color=config.gt_color, linewidth=config.line_width, zorder=5, alpha=0.6)  # 连接线

        # 获取对应物体的Result1轨迹的起始点
        if obj_id in result_filtered1:
            obj_result = result_filtered1[obj_id]
            first_result_point = obj_result[0, 3:5]  # Result的起始点
            plt.plot([last_point[0], first_result_point[0]], [last_point[1], first_result_point[1]], color=config.result_color1, linewidth=config.line_width, zorder=4, linestyle='--')  # 连接线

        if plot_second_result:
           # 获取对应物体的Result2轨迹的起始点
            if obj_id in result_filtered2:
                obj_result = result_filtered2[obj_id]
                first_result_point = obj_result[0, 3:5]  # Result的起始点
                plt.plot([last_point[0], first_result_point[0]], [last_point[1], first_result_point[1]], color=config.result_color2, linewidth=config.line_width, zorder=4, linestyle='--')  # 连接线

    # 绘制每个物体的result轨迹
    for obj_id in result_filtered1:
        obj_result = result_filtered1[obj_id]
        dot_shape = choose_shape(obj_result)
        plt.plot(obj_result[:, 3], obj_result[:, 4], color=config.result_color1, linewidth=config.line_width,  marker=dot_shape, markersize=config.dot_size,
                 markerfacecolor=config.result_color1, zorder=4, linestyle='--')  # 细线，绿色为预测轨迹

    if plot_second_result:
        # 绘制每个物体的result轨迹
        for obj_id in result_filtered2:
            obj_result = result_filtered2[obj_id]
            dot_shape = choose_shape(obj_result)
            plt.plot(obj_result[:, 3], obj_result[:, 4], color=config.result_color2, linewidth=config.line_width,
                     marker=dot_shape, markersize=config.dot_size,
                     markerfacecolor=config.result_color2, zorder=4, linestyle='--')  # 细线，绿色为预测轨迹

    # 绘制每个物体的Ground Truth轨迹
    for obj_id in gt_filtered:
        obj_gt = gt_filtered[obj_id]
        dot_shape = choose_shape(obj_gt)
        plt.plot(obj_gt[:, 3], obj_gt[:, 4], color=config.gt_color, linewidth=config.line_width, marker=dot_shape, markersize=config.dot_size,
                 markerfacecolor=config.gt_color, zorder=5, alpha=0.6)    # 细线，红色为GT轨迹


    # 创建一个自定义的图例项，仅包含一个点，标注为 'bike'
    legend_point_ped = Line2D([0], [0], marker='*', color='green', markerfacecolor='green', markersize=config.dot_size,
                          label='pedestrian')
    legend_point_bik = Line2D([0], [0], marker='d', color='gray', markerfacecolor='gray', markersize=config.dot_size,
                          label='bike')
    legend_point_veh = Line2D([0], [0], marker='o', color='brown', markerfacecolor='brown', markersize=config.dot_size,
                          label='vehicle')
    legend_point_oth = Line2D([0], [0], marker='s', color='red', markerfacecolor='red', markersize=config.dot_size,
                          label='otners')

    # 设置坐标轴刻度和网格
    # plt.grid(True)

    #
    # plt.xlim(-15, 15)
    plt.ylim(-90, 250)
    # 设置 X 轴和 Y 轴每 20 个单位一个刻度
    plt.xticks(np.arange(-15, 15 + 1, 5))
    plt.yticks(np.arange(-90, 250 + 1, 20))

    # 设置标题、标签
    # plt.title('Vehicle Trajectory Prediction')
    # plt.xlabel('X Position')
    # plt.ylabel('Y Position')

    # 添加图例
    history_line = plt.Line2D([0], [0], color='gray', lw=2, label='History')
    gt_line = plt.Line2D([0], [0], color='red', lw=2, label='Ground Truth')
    result_line = plt.Line2D([0], [0], color='green', lw=2, label='Prediction')

    # plt.legend(handles=[ legend_point_veh]) #history_line, gt_line, result_line,legend_point

    plt.show()





# 数据文件路径
history_filename = './history_v.txt'
gt_filename = './gt_v.txt'
method1_result_filename = './result_v.txt'
method2_result_filename = './prediction_result_0000.txt'


config = {
    'line_width': 2.5, # 3
    'history_color': 'gray',
    'gt_color': 'red',
    'result_color1': 'green',
    'result_color2': 'blue',
    'dot_size': 5,  # 6.5
    'fig_size':(3,13),

}
config = type('Config', (object,), config)()



# 读取数据
history_data = load_data(history_filename)
gt_data = load_data(gt_filename)
result_data1 = load_data(method1_result_filename)
result_data2 = load_data(method2_result_filename)

# 绘制轨迹
plot_trajectory(history_data, gt_data, result_data1, result_data2, plot_second_result=True)
