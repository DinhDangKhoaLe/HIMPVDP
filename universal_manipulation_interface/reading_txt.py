import zarr
import zipfile
import tempfile
import h5py    
import numpy as np  
# import imagecodecs
import pickle
import glob
import matplotlib.pyplot as plt
import os

DATA_PATH = '/home/augustine/lfd_ws/src/HIMPVDP/universal_manipulation_interface/Data_ICRA/'
DATA_FOLDER_PRE_UMI = 'PRE_UMI' 
DATA_FOLDER_UMI = 'UMI'
COUNTER_UMI = 1
COUNTER_PRE_UMI = 5

def find_movement_indices(hand_positions, threshold=0.02):
    # Calculate the change in hand position between consecutive time steps
    hand_position_changes = np.linalg.norm(np.diff(hand_positions, axis=0), axis=1)
    
    start_index = None
    stop_index = None
    moving = False
    
    for i, change in enumerate(hand_position_changes):
        print(i, change)
        if change > threshold and not moving:
            start_index = i
            moving = True
        elif change < threshold and moving:
            stop_index = i
            moving = False
            break  # Stop after finding the first stop index
    
    return start_index, stop_index

# how can use reprix
def read_text_file(file_path, prefix):
    data = []
    # Open the text file and read line by line
    with open(file_path, 'r') as file:
        if prefix == 'Arm_pos:' or 'Hand_pos':
            for line in file:
                # Look for lines that start with the given prefix
                if line.startswith(prefix):
                    # Extract the numerical values inside the square brackets
                    values = line.split('[')[1].split(']')[0]
                    # Convert the values to a list of floats
                    position = list(map(float, values.split(',')))
                    data.append(position)
        elif prefix == 'Distance:' or 'Duration:':
            for line in file:
                # Look for lines that start with the given prefix
                if line.startswith(prefix):
                    # Extract the numerical values inside the square brackets
                    values = line.split('[')[1].split(']')[0]
                    # Convert the values to a list of floats
                    position = list(map(float, values.split(',')))
                    data.extend(position)
    data_array = np.array(data)
    return data_array

def main():
    umi_data_dict = {}
    our_approach_data_dict = {}
    
    # datapath = os.path.join(DATA_PATH + DATA_FOLDER_UMI, f"*_{COUNTER}.txt")
    files_umi = glob.glob(os.path.join(DATA_PATH + DATA_FOLDER_UMI, f"*_{COUNTER_UMI}.txt"))
    file_our_approach = glob.glob(os.path.join(DATA_PATH + DATA_FOLDER_PRE_UMI, f"*_{COUNTER_PRE_UMI}.txt"))
    
    for file_path in files_umi:
        if '/pre_hand_pos' in file_path:
            umi_data_dict['pre_hand_positions'] = read_text_file(file_path, 'Hand_pos:')
        if '/hand_pos' in file_path:
            umi_data_dict['hand_positions'] = read_text_file(file_path, 'Hand_pos:')
        if '/arm_pos' in file_path:
            umi_data_dict['arm_positions'] = read_text_file(file_path, 'Arm_pos:')
        if '/distance' in file_path:
            umi_data_dict['distances'] = read_text_file(file_path, 'Distance:')
        if '/duration' in file_path:
            umi_data_dict['durations'] = read_text_file(file_path, 'Duration:')
    
    for file_path in file_our_approach:
        if '/pre_hand_pos' in file_path:
            our_approach_data_dict['pre_hand_positions'] = read_text_file(file_path, 'Hand_pos:')
        if '/hand_pos' in file_path:
            our_approach_data_dict['hand_positions'] = read_text_file(file_path, 'Hand_pos:')
        if '/arm_pos' in file_path:
            our_approach_data_dict['arm_positions'] = read_text_file(file_path, 'Arm_pos:')
        if '/distance' in file_path:
            our_approach_data_dict['distances'] = read_text_file(file_path, 'Distance:')
        if '/duration' in file_path:
            our_approach_data_dict['durations'] = read_text_file(file_path, 'Duration:')

    # Trim n data points at the end for UMI
    m1 = 0  
    n1 = 1

    # Trim n data points at the end for UMI_pre
    m2 = 0
    n2 = 2

    for key in ['distances', 'durations']:
        if key in umi_data_dict:
            umi_data_dict[key] = umi_data_dict[key].reshape(-1)
            umi_data_dict[key] = umi_data_dict[key][m1:-n1]
        if key in our_approach_data_dict:
            our_approach_data_dict[key] = our_approach_data_dict[key].reshape(-1)
            our_approach_data_dict[key] = our_approach_data_dict[key][m2:-n2]

    for key in ['pre_hand_positions', 'hand_positions', 'arm_positions']:
        if key in umi_data_dict:
            umi_data_dict[key] = umi_data_dict[key][m1:-n1]
        if key in our_approach_data_dict:
            our_approach_data_dict[key] = our_approach_data_dict[key][m2:-n2]
    
    # Calculate the distance to goal for each hand position
    umi_data_dict['goal_positions'] = umi_data_dict['hand_positions'][-1]
    umi_data_dict['hand_distances_to_goal'] = np.linalg.norm(umi_data_dict['hand_positions'] - umi_data_dict['goal_positions'], axis=1)
    umi_data_dict['arm_distances_to_goal'] = np.linalg.norm(umi_data_dict['arm_positions'] - umi_data_dict['goal_positions'], axis=1)

    our_approach_data_dict['goal_positions'] = our_approach_data_dict['hand_positions'][-1]
    our_approach_data_dict['hand_distances_to_goal'] = np.linalg.norm(our_approach_data_dict['hand_positions'] - our_approach_data_dict['goal_positions'], axis=1)
    our_approach_data_dict['arm_distances_to_goal'] = np.linalg.norm(our_approach_data_dict['arm_positions'] - our_approach_data_dict['goal_positions'], axis=1)

    array_time_umi = np.array([5.36, 5.96, 4.16, 5.36])
    array_time_approach = np.array([2.16, 2.16, 2.46, 2.76])
    # Calculate the mean and standard deviation of the array
    array_mean_umi = np.mean(array_time_umi)
    array_std_umi = np.std(array_time_umi)

    array_mean_approach = np.mean(array_time_approach)
    array_std_approach = np.std(array_time_approach)

    # Print the mean and standard deviation
    print("Mean:", array_mean_umi)
    print("Standard Deviation:", array_std_umi)

    print("Mean:", array_mean_approach)
    print("Standard Deviation:", array_std_approach)

    plt.rcParams['lines.linewidth'] = 5.0

    
    # Plot the hand trajectory
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.xlim(0.35, 5.5)
    plt.plot(umi_data_dict['durations'], umi_data_dict['hand_distances_to_goal'], label='Hand Distance to Goal')
    plt.plot(umi_data_dict['durations'], umi_data_dict['arm_distances_to_goal'], label='Arm Distance To Goal', color='red')
    plt.plot([umi_data_dict['durations'][0], umi_data_dict['durations'][-5]], [umi_data_dict['arm_distances_to_goal'][0], umi_data_dict['hand_distances_to_goal'][-5]], label='Ideal Trajectory', color='green', linestyle='--')
    # plt.axhline(y=0, color='black', linestyle='-', label='Goal')
    plt.fill_between(x=[umi_data_dict['durations'][0], umi_data_dict['durations'][-1]], y1=0, y2=0.1, color='black', alpha=0.1, label='Goal Range')

    plt.legend(fontsize=20)
    plt.xlabel('Time (seconds)', fontsize=16)
    plt.ylabel('Distance to Goal (m)', fontsize=16)
    plt.title('VDP',fontweight='bold', fontsize=18)
    plt.legend()
    plt.xticks(fontsize=14)  # Set x-axis tick labels font size
    plt.yticks(fontsize=14)  # Set y-axis tick labels font size

    plt.subplot(1, 2, 2)
    plt.plot(our_approach_data_dict['durations'], our_approach_data_dict['hand_distances_to_goal'], label='Hand Distance to Goal')
    plt.plot(our_approach_data_dict['durations'], our_approach_data_dict['arm_distances_to_goal'], label='Arm Distance To Goal', color='red')
    plt.plot([our_approach_data_dict['durations'][0], our_approach_data_dict['durations'][-3]], [our_approach_data_dict['arm_distances_to_goal'][0], our_approach_data_dict['hand_distances_to_goal'][-3]], label='Ideal Trajectory', color='green', linestyle='--')
    
    # plt.axhline(y=0, color='black', linestyle='-', label='Goal')
    plt.fill_between(x=[our_approach_data_dict['durations'][0], our_approach_data_dict['durations'][-1]], y1=0, y2=0.1, color='black', alpha=0.1, label='Goal Range')

    plt.legend(fontsize=20)
    plt.xlabel('Time (seconds)', fontsize=16)
    plt.ylabel('Distance to Goal (m))', fontsize=16)
    plt.title('VDP + Hand Prediction',fontweight='bold', fontsize=18)
    plt.legend()
    plt.xticks(fontsize=14)  # Set x-axis tick labels font size
    plt.yticks(fontsize=14)  # Set y-axis tick labels font size

    # Plot distance over duration for each data dictionary
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(umi_data_dict['durations'], umi_data_dict['distances'])
    plt.xlabel('Time (seconds)', fontsize=16)
    plt.ylabel('Distance (m)', fontsize=16)
    plt.title('VDP',fontweight='bold', fontsize=18)
    plt.xticks(fontsize=14)  # Set x-axis tick labels font size
    plt.yticks(fontsize=14)  # Set y-axis tick labels font size

    plt.subplot(1, 2, 2)
    plt.plot(our_approach_data_dict['durations'], our_approach_data_dict['distances'])
    plt.xlabel('Time (seconds)', fontsize=16)
    plt.ylabel('Distance (m)', fontsize=16)
    plt.title('VDP + Hand Prediction',fontweight='bold', fontsize=18)
    plt.xticks(fontsize=14)  # Set x-axis tick labels font size
    plt.yticks(fontsize=14)  # Set y-axis tick labels font size
    
    plt.tight_layout()
    plt.show()
    # # Plot arm position over duration in 3D
    # # Plot hand and arm position on the same figure with different colors

    # fig = plt.figure()
    # ax = plt.axes(projection='3d')
    # # ax.axes.set_xlim3d(left=0, right=1.5) 
    # # ax.axes.set_ylim3d(bottom=-1, top=0) 
    # ax.axes.set_zlim3d(bottom=0.3, top=1.0) 
    # ax.plot3D(hand_positions[:, 0], hand_positions[:, 1], hand_positions[:, 2], color='red', label='Hand Position')
    # ax.plot3D(arm_positions[:, 0], arm_positions[:, 1], arm_positions[:, 2], color='blue', label='Arm Position')
    # ax.legend()
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # ax.set_title('Arm Position over Duration (3D)')
    # plt.show()

    # # Plot distance over duration
    # plt.figure()
    # plt.plot(durations, distances)
    # plt.xlabel('Duration')
    # plt.ylabel('Distance')
    # plt.title('Distance over Duration')
    # plt.show()

if __name__ == '__main__':
    main()