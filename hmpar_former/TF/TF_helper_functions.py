import torch
import torch.nn as nn
import numpy as np


# Generate sequences for training and testing
def generate_sequences(norm_pos, norm_vel, norm_acc, input_length=60, predict_length=60):
    num_frames = norm_pos.shape[0]
    num_joints = norm_pos.shape[1]

    # Calculate the total number of sequences we can create
    num_sequences = num_frames - input_length - predict_length + 1

    # Initialize arrays to store the input and target sequences
    X_pos = np.zeros((num_sequences, input_length, num_joints, 3))
    X_vel = np.zeros((num_sequences, input_length, num_joints, 3))
    X_acc = np.zeros((num_sequences, input_length, num_joints, 3))
    Y_pos = np.zeros((num_sequences, predict_length, num_joints, 3))
    Y_vel = np.zeros((num_sequences, predict_length, num_joints, 3))
    Y_acc = np.zeros((num_sequences, predict_length, num_joints, 3))

    # Create sequences
    for i in range(num_sequences):
        X_pos[i] = norm_pos[i:i + input_length]
        X_vel[i] = norm_vel[i:i + input_length]
        X_acc[i] = norm_acc[i:i + input_length]
        Y_pos[i] = norm_pos[i + input_length:i + input_length + predict_length]
        Y_vel[i] = norm_vel[i + input_length:i + input_length + predict_length]
        Y_acc[i] = norm_acc[i + input_length:i + input_length + predict_length]

    return X_pos, X_vel, X_acc, Y_pos, Y_vel, Y_acc

# Create a mask to prevent the missing joints
def create_shifted_mask(seq_length, num_joints):
    # seq_length is the number of time steps
    # num_joints is the number of joints per time step
    total_length = seq_length * num_joints
    mask = torch.ones((total_length, total_length), dtype=torch.float32) * float('-inf')  # Start with everything masked
    for i in range(seq_length):
        for j in range(i + 1):  # Allow visibility up to and including the current time step
            start_row = i * num_joints
            end_row = start_row + num_joints
            start_col = j * num_joints
            end_col = start_col + num_joints
            mask[start_row:end_row, start_col:end_col] = 0.0  # Unmask the allowed region
    return mask

# Masked Mean Squared Error Loss
class MaskedMSELoss(nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()

    def forward(self, output, target):
        squared_diff = (output - target) ** 2
        loss = squared_diff.mean()
        return loss

# Reverse normalization of the data using the median and IQR
def reverse_normalization(normalized_data, medians_per_joint_axis, iqrs_per_joint_axis):
    original_data = np.empty_like(normalized_data)  # Initialize an array to hold the original data

    # Iterate over each joint and each axis
    for joint in range(normalized_data.shape[0]):
        for axis in range(normalized_data.shape[1]):
            # Retrieve the median and IQR for this joint and axis
            median = medians_per_joint_axis[joint, axis]
            iqr = iqrs_per_joint_axis[joint, axis]

            # Retrieve the normalized values for this joint and axis
            normalized_values = normalized_data[joint, axis]

            # Calculate the original values based on the normalization formula
            original_values = (normalized_values * iqr) + median

            # Store the original values in the output array
            original_data[joint, axis] = original_values

    return original_data

# Normalize the data using the median and IQR
def robust_normalize_data_with_clipping(data, medians_per_joint_axis, iqrs_per_joint_axis, normalized_data, clipping_percentiles=(1, 99)):
    for joint in range(data.shape[1]):  # For each joint
        for axis in range(data.shape[2]):  # For each axis (x, y, z)
            joint_axis_data = data[:, joint, axis]
            # Determine clipping thresholds based on percentiles
            lower_threshold, upper_threshold = np.percentile(joint_axis_data, clipping_percentiles)
            # Clip the data based on thresholds
            clipped_values = np.clip(joint_axis_data, lower_threshold, upper_threshold)
            # Normalize the clipped data, avoiding division by zero
            if iqrs_per_joint_axis[joint, axis] > 0:
                normalized_values = (clipped_values - medians_per_joint_axis[joint, axis]) / iqrs_per_joint_axis[joint, axis]
            else:
                normalized_values = clipped_values  # Keep original values if IQR is 0
            # Store the normalized values
            normalized_data[:, joint, axis] = normalized_values
    return normalized_data