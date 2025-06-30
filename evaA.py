'''
Evaluation code for trajectory prediction.

We record the objects in the last frame of every sequence in test dataset as considered objects, which is stored in considered_objects.txt.
We compare the error between your predicted locations in the next 3s(six positions) and the ground truth for these considered objects.

To run this script, make sure that your results are in required format.
'''

import os
import argparse
import numpy as np
import glob

from sympy.codegen.cnodes import sizeof


def evaluation(frame_data_result, frame_data_gt, consider_peds):
    """
    annotated objects in Apollo train data
    1: 25373 0.36
    2: 5572 0.08
    3: 18043 0.25
    4: 11417 0.16
    5: 10792 0.15
    """
    # defined length of predicted trajectory
    predict_len = 6
    # the counter for testing sequences
    sequence_count = 0
    # weighted coefficient for vehicles (1,2), pedestrians (3), bicyclists(4) respectively, objects annotated as 5 is omitted
    vehicle_coe = 0.2
    pedestrian_coe = 0.58
    bicycle_coe = 0.22
    # error for missing considered objects
    miss_error = 100
    # record displacement error for three types of objects
    vehicle_error = []
    pedestrian_error = []
    bicycle_error = []
    # record final displacement error for three types of objects
    vehicle_final_error = []
    pedestrian_final_error = []
    bicycle_final_error = []
    # record RMSE for vehicles at different prediction horizons
    vehicle_error_1frame = []
    vehicle_error_2frame = []
    vehicle_error_3frame = []
    vehicle_error_4frame = []
    vehicle_error_5frame = []
    vehicle_error_6frame = []
    RMSE = []



    for i in range(0, len(frame_data_result) - predict_len + 1, predict_len):
        current_consider_ped = list(map(int, consider_peds[sequence_count]))  # 转换为列表
        sequence_count += 1
        for j in range(i, i + predict_len):
            prediction_horizon = j-i+1
            for ped_gt in frame_data_gt[j]:
                if int(ped_gt[0]) in current_consider_ped:  # 使用 in 代替 count
                    # ignore unknown objects
                    if ped_gt[1] == 5:
                        continue
                    # error will be large if missing considered objects
                    error = miss_error
                    for ped_res in frame_data_result[j]:
                        if int(ped_res[0]) == int(ped_gt[0]):
                            error = distance([ped_gt[2], ped_gt[3]], [ped_res[2], ped_res[3]])
                            break
                    # distribute the error to different types of objects
                    if ped_gt[1] == 1 or ped_gt[1] == 2:
                        vehicle_error.append(error)
                        # print(f"prediction frame: {prediction_horizon}")
                        # print(error)
                        if prediction_horizon ==1:
                            vehicle_error_1frame.append(error)
                        elif prediction_horizon ==2:
                            vehicle_error_2frame.append(error)
                        elif prediction_horizon ==3:
                            vehicle_error_3frame.append(error)
                        elif prediction_horizon ==4:
                            vehicle_error_4frame.append(error)
                        elif prediction_horizon ==5:
                            vehicle_error_5frame.append(error)
                        elif prediction_horizon ==6:
                            vehicle_error_6frame.append(error)





                        if j == i + predict_len - 1:
                            vehicle_final_error.append(error)
                    elif ped_gt[1] == 3:
                        pedestrian_error.append(error)
                        if j == i + predict_len - 1:
                            pedestrian_final_error.append(error)
                    elif ped_gt[1] == 4:
                        bicycle_error.append(error)
                        if j == i + predict_len - 1:
                            bicycle_final_error.append(error)

    # calculate RMSE
    vehicle_errors = [
        vehicle_error_1frame, vehicle_error_2frame, vehicle_error_3frame,
        vehicle_error_4frame, vehicle_error_5frame, vehicle_error_6frame
    ]

    for i, error in enumerate(vehicle_errors, start=1):
        rmse_value = round(np.sqrt(np.mean(np.square(error))), 4)
        RMSE.append(rmse_value)
        print(f'{i} frame RMSE for vehicle: {rmse_value}')

    # the mean error for objects
    vehicle_mean_error = sum(vehicle_error) / len(vehicle_error)
    pedestrian_mean_error = sum(pedestrian_error) / len(pedestrian_error)
    bicycle_mean_error = sum(bicycle_error) / len(bicycle_error)
    # the final error for objects
    vehicle_final_error = sum(vehicle_final_error) / len(vehicle_final_error)
    pedestrian_final_error = sum(pedestrian_final_error) / len(pedestrian_final_error)
    bicycle_final_error = sum(bicycle_final_error) / len(bicycle_final_error)
    # weighted sum of mean error
    WSADE = vehicle_mean_error * vehicle_coe + pedestrian_mean_error * pedestrian_coe + bicycle_mean_error * bicycle_coe
    # weighted sum of final error
    WSFDE = vehicle_final_error * vehicle_coe + pedestrian_final_error * pedestrian_coe + bicycle_final_error * bicycle_coe

    print(f'WSADE: {WSADE:.4f}')  # 保留4位小数
    print(f'ADEv: {vehicle_mean_error:.4f}, ADEp: {pedestrian_mean_error:.4f}, ADEb: {bicycle_mean_error:.4f}')
    print(f'WSFDE: {WSFDE:.4f}')  # 保留4位小数
    print(f'FDEv: {vehicle_final_error:.4f}, FDEp: {pedestrian_final_error:.4f}, FDEb: {bicycle_final_error:.4f}')
    print("\n")

    return (WSADE, vehicle_mean_error, pedestrian_mean_error, bicycle_mean_error,
            WSFDE, vehicle_final_error, pedestrian_final_error, bicycle_final_error, RMSE)


def readConsiderObjects(filename):
    # print('Load file: ', filename)

    # load considered objects of each sequence
    consider_peds = []
    with open(filename, 'r') as file_to_read:
        while True:
            lines = file_to_read.readline()
            if not lines:
                break

            curLine = lines.strip().split(" ")
            intLine = map(int, curLine)
            consider_peds.append(intLine)

    return consider_peds


def readTrajectory(filename):

    # print('Load file: ',filename)
    raw_data = []
    # load all the data in the file
    with open(filename, 'r') as file_to_read:
        while True:
            lines = file_to_read.readline()
            if not lines:
                break
            timestamp, id, type, x, y = [float(i) for i in lines.split()]
            raw_data.append((timestamp, id, type, x, y))

    # get frame list
    frameList = []
    for i in range(len(raw_data)):
        if frameList.count(raw_data[i][0]) == 0:
            frameList.append(raw_data[i][0])

    counter = 0
    frame_data = []
    for ind, frame in enumerate(frameList):
        pedsInFrame = []
        # Extract all pedestrians in current frame
        for r in range(counter, len(raw_data)):
            row = raw_data[r]

            if raw_data[r][0] == frame:
                pedsInFrame.append([row[1], row[2], row[3], row[4]])
                counter += 1
            else:
                break

        frame_data.append(pedsInFrame)

    return frame_data


def distance(pos1, pos2):
    # Euclidean distance
    return np.sqrt(pow(pos1[0]-pos2[0], 2) + pow(pos1[1]-pos2[1], 2))


def main():
    parser = argparse.ArgumentParser(
        description='Evaluation self localization.')

    parser.add_argument('--gt_dir', default='test_eval_data/Apollo_prediction_gt.txt',
                        help='the dir of ground truth')
    parser.add_argument('--object_file', default='test_eval_data/Apollo_considered_objects.txt',
                        help='the dir of considered objects')
    parser.add_argument('--res_dir', default='test_eval_data/Apollo_prediction_result_refine2_200',
                        help='the dir of results')

    args = parser.parse_args()

    # load ground truth
    file_gt = args.gt_dir
    frame_data_gt = readTrajectory(file_gt)



    # load prediction results
    dir_result = args.res_dir
    log_file = os.path.join(dir_result, 'test_result')
    frame_data_result = sorted(glob.glob(os.path.join(dir_result, '*.txt')))  # 获取所有txt文件路径并排序

    # Open the log file to write results
    with open(log_file, 'w') as writer:
        for files in frame_data_result:

            # load considered objects
            file_consider_objects = args.object_file
            consider_peds = readConsiderObjects(file_consider_objects)

            frame_data_result_data = readTrajectory(files)  # 读取每个预测文件的数据
            # Do evaluation and capture the results
            WSADE, vehicle_mean_error, pedestrian_mean_error, bicycle_mean_error, WSFDE, vehicle_final_error, pedestrian_final_error, bicycle_final_error, RMSE = evaluation(
                frame_data_result_data, frame_data_gt, consider_peds)

            # Write results to the log file in the desired format
            writer.write(f"Processing file: {files}\n")
            writer.write(" ".join(map(str, RMSE)) + "\n")

            writer.write(f"WSADE: {WSADE:.4f}\n")
            writer.write(
                f"ADEv: {vehicle_mean_error:.4f}, ADEp: {pedestrian_mean_error:.4f}, ADEb: {bicycle_mean_error:.4f}\n")
            writer.write(f"WSFDE: {WSFDE:.4f}\n")
            writer.write(
                f"FDEv: {vehicle_final_error:.4f}, FDEp: {pedestrian_final_error:.4f}, FDEb: {bicycle_final_error:.4f}\n")
            writer.write("\n")  # Add an empty line between results for each file

    print("Evaluation results have been saved")

if __name__ == '__main__':

    main()



