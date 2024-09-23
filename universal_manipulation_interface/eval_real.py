# %%
import os
import pathlib
import time
from multiprocessing.managers import SharedMemoryManager

import av
import click
import cv2
import yaml
import dill
import hydra
import numpy as np
import scipy.spatial.transform as st
import torch
import copy
import sys
from omegaconf import OmegaConf
import json
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.cv2_util import (
    get_image_transform
)
from umi.common.cv_util import (
    parse_fisheye_intrinsics,
    FisheyeRectConverter
)
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from umi.common.precise_sleep import precise_wait
from umi.real_world.bimanual_umi_env import BimanualUmiEnv
from umi.real_world.keystroke_counter import (
    KeystrokeCounter, Key, KeyCode
)
from umi.real_world.real_inference_util import (get_real_obs_dict,
                                                get_real_obs_resolution,
                                                get_real_umi_obs_dict,
                                                get_real_umi_action)
from umi.common.pose_util import pose_to_mat, mat_to_pose
# from ultralytics import YOLO
import matplotlib.pyplot as plt

module_path = os.path.abspath('/home/augustine/lfd_ws/src/HIMPVDP/hmpar_former/TF')
if module_path not in sys.path:
    sys.path.append(module_path)

import realtime_inference_lstm as zed_inference_lstm

from scipy.spatial.transform import Rotation as R
import os

OmegaConf.register_new_resolver("eval", eval, replace=True)


DATA_PATH = '/home/augustine/lfd_ws/src/HIMPVDP/universal_manipulation_interface/Data_ICRA/'
# DATA_FOLDER = 'UMI/' # change this to use the umi
DATA_FOLDER = 'PRE_UMI/' # change this to use umi with prediction
COUNTER = 5

DISTANCE_THRESHOLD_MAX = 0.25
DISTANCE_THRESHOLD_MIN = 0.1
ALPHA_MAX = 1.0
ALPHA_MIN = 0.0


def compute_rotation_vector(current_position, current_orientation, target_position):
    # Convert inputs to numpy arrays
    current_position = np.array(current_position)
    target_position = np.array(target_position)
    current_orientation = np.array(current_orientation)

    # Convert the current orientation from rotation vector to rotation matrix
    current_rotation = R.from_rotvec(current_orientation)
    current_orientation_matrix = current_rotation.as_matrix()

    # Compute the target direction vector (from current position to target position)
    v_target = target_position - current_position
    v_target = v_target / np.linalg.norm(v_target)  # Normalize

    # Extract the current facing direction (z-axis of the current_orientation matrix)
    v_current = current_orientation_matrix[:, 2]  # End-effector's forward direction in base frame
    v_current = v_current / np.linalg.norm(v_current)  # Normalize

    # Compute the rotation axis and angle
    rotation_axis = np.cross(v_current, v_target)
    if np.linalg.norm(rotation_axis) < 1e-6:
        # If the rotation axis is zero, the vectors are collinear (no rotation needed)
        rotation_axis = np.array([1, 0, 0])  # Arbitrary axis
    else:
        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)  # Normalize

    # Compute the rotation angle
    rotation_angle = np.arccos(np.clip(np.dot(v_current, v_target), -1.0, 1.0))

    # Create the rotation matrix
    rotation = R.from_rotvec(rotation_angle * rotation_axis)
    rotation_matrix = rotation.as_matrix()

    # Compute the new orientation matrix
    new_orientation_matrix = np.dot(rotation_matrix, current_orientation_matrix)

    # Convert the new orientation matrix back to a rotation vector
    new_rotation = R.from_matrix(new_orientation_matrix)
    rotation_vector = new_rotation.as_rotvec()
    return rotation_vector

@click.command()
@click.option('--input', '-i', required=True, help='Path to checkpoint')
@click.option('--output', '-o', required=True, help='Directory to save recording')
@click.option('--robot_config', '-rc', required=True, help='Path to robot_config yaml file')
@click.option('--match_dataset', '-m', default=None, help='Dataset used to overlay and adjust initial condition')
@click.option('--match_episode', '-me', default=None, type=int, help='Match specific episode from the match dataset')
@click.option('--match_camera', '-mc', default=0, type=int)
@click.option('--camera_reorder', '-cr', default='0')
@click.option('--vis_camera_idx', default=0, type=int, help="Which RealSense camera to visualize.")
@click.option('--init_joints', '-j', is_flag=True, default=False, help="Whether to initialize robot joint configuration in the beginning.")
@click.option('--steps_per_inference', '-si', default=3, type=int, help="Action horizon for inference.")
@click.option('--max_duration', '-md', default=2000000, help='Max duration for each epoch in seconds.')
@click.option('--frequency', '-f', default=10, type=float, help="Control frequency in Hz.")
@click.option('--command_latency', '-cl', default=0.01, type=float, help="Latency between receiving SapceMouse command to executing on Robot in Sec.")
@click.option('-nm', '--no_mirror', is_flag=True, default=False)
@click.option('-sf', '--sim_fov', type=float, default=None)
@click.option('-ci', '--camera_intrinsics', type=str, default=None)
@click.option('--mirror_swap', is_flag=True, default=False)
def main(input, output, robot_config, 
    match_dataset, match_episode, match_camera,
    camera_reorder,
    vis_camera_idx, init_joints, 
    steps_per_inference, max_duration,
    frequency, command_latency, 
    no_mirror, sim_fov, camera_intrinsics, mirror_swap):
    max_gripper_width = 0.09
    gripper_speed = 0.2
    
    # load robot config file
    robot_config_data = yaml.safe_load(open(os.path.expanduser(robot_config), 'r'))
    
    # load left-right robot relative transform
    tx_left_right = np.array(robot_config_data['tx_left_right'])
    tx_robot1_robot0 = tx_left_right
    
    robots_config = robot_config_data['robots']
    grippers_config = robot_config_data['custom_grippers']

    # load checkpoint
    ckpt_path = input
    if not ckpt_path.endswith('.ckpt'):
        ckpt_path = os.path.join(ckpt_path, 'checkpoints', 'latest.ckpt')
    payload = torch.load(open(ckpt_path, 'rb'), map_location='cpu', pickle_module=dill)
    cfg = payload['cfg']
    print("model_name:", cfg.policy.obs_encoder.model_name)
    print("dataset_path:", cfg.task.dataset.dataset_path)

    # setup experiment
    dt = 1/frequency

    obs_res = get_real_obs_resolution(cfg.task.shape_meta)
    # load fisheye converter
    fisheye_converter = None
    if sim_fov is not None:
        assert camera_intrinsics is not None
        opencv_intr_dict = parse_fisheye_intrinsics(
            json.load(open(camera_intrinsics, 'r')))
        fisheye_converter = FisheyeRectConverter(
            **opencv_intr_dict,
            out_size=obs_res,
            out_fov=sim_fov
        )

    print("steps_per_inference:", steps_per_inference)
    with SharedMemoryManager() as shm_manager:
        with KeystrokeCounter() as key_counter, \
            BimanualUmiEnv(
                output_dir=output,
                robots_config=robots_config,
                grippers_config=grippers_config,
                frequency=frequency,
                obs_image_resolution=obs_res,
                obs_float32=True,
                camera_reorder=[int(x) for x in camera_reorder],
                init_joints=init_joints,
                enable_multi_cam_vis=False,
                # latency
                camera_obs_latency=0.117, # 0.17 before
                # obs
                camera_obs_horizon=cfg.task.shape_meta.obs.camera0_rgb.horizon,
                robot_obs_horizon=cfg.task.shape_meta.obs.robot0_eef_pos.horizon,
                gripper_obs_horizon=cfg.task.shape_meta.obs.robot0_gripper_width.horizon,
                no_mirror=no_mirror,
                fisheye_converter=fisheye_converter,
                mirror_swap=mirror_swap,
                # action
                max_pos_speed=0.09, #0.08 for pre 0.05 for umi
                max_rot_speed=0.4, #0.3 for pre and 0.1 for umi
                shm_manager=shm_manager) as env:
            cv2.setNumThreads(2)
            print("Waiting for camera")
            time.sleep(1.0) 

            # Setup Zed Camera
            zed_prediction = zed_inference_lstm.Realtime_Inference()

            # load match_dataset
            episode_first_frame_map = dict()
            match_replay_buffer = None
            if match_dataset is not None:
                match_dir = pathlib.Path(match_dataset)
                match_zarr_path = match_dir.joinpath('replay_buffer.zarr')
                match_replay_buffer = ReplayBuffer.create_from_path(str(match_zarr_path), mode='r')
                match_video_dir = match_dir.joinpath('videos')
                for vid_dir in match_video_dir.glob("*/"):
                    episode_idx = int(vid_dir.stem)
                    match_video_path = vid_dir.joinpath(f'{match_camera}.mp4')
                    if match_video_path.exists():
                        img = None
                        with av.open(str(match_video_path)) as container:
                            stream = container.streams.video[0]
                            for frame in container.decode(stream):
                                img = frame.to_ndarray(format='rgb24')
                                break
                        episode_first_frame_map[episode_idx] = img
            print(f"Loaded initial frame for {len(episode_first_frame_map)} episodes")

            # creating model
            # have to be done after fork to prevent 
            # duplicating CUDA context with ffmpeg nvenc
            cls = hydra.utils.get_class(cfg._target_)
            workspace = cls(cfg)
            workspace: BaseWorkspace
            workspace.load_payload(payload, exclude_keys=None, include_keys=None)

            policy = workspace.model
            if cfg.training.use_ema:
                policy = workspace.ema_model
            policy.num_inference_steps = 16 # DDIM inference iterations
            obs_pose_rep = cfg.task.pose_repr.obs_pose_repr
            action_pose_repr = cfg.task.pose_repr.action_pose_repr

            device = torch.device('cuda')
            policy.eval().to(device)

            print("Warming up policy inference")
            obs = env.get_obs()
            episode_start_pose = list()
            for robot_id in range(len(robots_config)):
                pose = np.concatenate([
                    obs[f'robot{robot_id}_eef_pos'],
                    obs[f'robot{robot_id}_eef_rot_axis_angle']
                ], axis=-1)[-1]
                episode_start_pose.append(pose)
            with torch.no_grad():
                policy.reset()
                obs_dict_np = get_real_umi_obs_dict(
                    env_obs=obs, shape_meta=cfg.task.shape_meta, 
                    obs_pose_repr=obs_pose_rep,
                    tx_robot1_robot0=tx_robot1_robot0,
                    episode_start_pose=episode_start_pose)
                obs_dict = dict_apply(obs_dict_np, 
                    lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
                result = policy.predict_action(obs_dict)
                action = result['action_pred'][0].detach().to('cpu').numpy()
                assert action.shape[-1] == 10 * len(robots_config)
                action = get_real_umi_action(action, obs, action_pose_repr)
                assert action.shape[-1] == 7 * len(robots_config)
                del result

            # for plotting
            # fig = plt.figure()
            # ax = plt.axes(projection='3d')
            # ax.axes.set_xlim3d(left=0, right=1.5) 
            # ax.axes.set_ylim3d(bottom=-1, top=0) 
            # ax.axes.set_zlim3d(bottom=0.0, top=1.0) 
            distance_vector = []
            duration_vector = []
            arm_pos_vector = []
            hand_pos_vector = []
            pre_hand_pos_vector = []
            
            print('Ready!')
            while True:
                t_start = time.monotonic()
                iter_idx = 0
                
                # ========== policy control loop ==============
                try:
                    # start episode
                    policy.reset()
                    start_delay = 1.0
                    eval_t_start = time.time() + start_delay
                    t_start = time.monotonic() + start_delay
                    env.start_episode(eval_t_start)

                    # get current pose
                    obs = env.get_obs()

                    episode_start_pose = list()
                    for robot_id in range(len(robots_config)):
                        pose = np.concatenate([
                            obs[f'robot{robot_id}_eef_pos'],
                            obs[f'robot{robot_id}_eef_rot_axis_angle']
                        ], axis=-1)[-1]
                        episode_start_pose.append(pose)

                    # wait for 1/30 sec to get the closest frame actually
                    # reduces overall latency
                    frame_latency = 1/30
                    precise_wait(eval_t_start - frame_latency, time_func=time.time)
                    print("Started!")
                    iter_idx = 0

                    
                    while True:
                        ##################### COMBINATION METHOD
                        # get obs 
                        obs = env.get_obs()
                        obs_timestamps = obs['timestamp']

                        # get current pose and future pose from hand prediction [x,y,z]
                        arm_position = obs[f'robot{robot_id}_eef_pos'][-1]
                        arm_rotation_vector= obs[f'robot{robot_id}_eef_rot_axis_angle'][-1]
                        
                        hand_position = np.array(zed_prediction.position)
                        goal_position = np.array(zed_prediction.predicted_position)
                        goal_rotation_vector = compute_rotation_vector(arm_position, arm_rotation_vector, goal_position)
                        distance = np.linalg.norm(arm_position - hand_position)

                        if np.isnan(hand_position).any() or np.all(hand_position == 0):
                            print("NO HUMAN HAND DETECTED")
                            continue
                        else:
                            if distance >= DISTANCE_THRESHOLD_MAX:
                                alpha = ALPHA_MIN
                            else:
                                if distance <= DISTANCE_THRESHOLD_MIN:
                                    alpha = ALPHA_MAX
                                else:
                                    alpha = ALPHA_MAX * (1 - (distance / DISTANCE_THRESHOLD_MAX-DISTANCE_THRESHOLD_MIN) ** 2)
                            print(f"Alpha: {alpha}")

                        # alpha = 1.0 # for now just use the hand prediction. Delete this line to use the combination method
                        if alpha == ALPHA_MIN:
                            t_cycle_end = t_start + (iter_idx+1)* dt
                            this_target_poses = np.concatenate((goal_position, goal_rotation_vector), axis=-1)
                            # env.robots[0].servoL(this_target_poses,0.1) #reach the goal in 0.1 second
                            precise_wait(t_cycle_end)
                            iter_idx += 1
                        else: 
                            # calculate timing
                            t_cycle_end = t_start + (iter_idx + steps_per_inference) * dt

                            # get obs 
                            obs = env.get_obs()
                            obs_timestamps = obs['timestamp']
                            # print(f'Obs latency {time.time() - obs_timestamps[-1]}')

                            # run inference
                            with torch.no_grad():
                                s = time.time()
                                obs_dict_np = get_real_umi_obs_dict(
                                    env_obs=obs, shape_meta=cfg.task.shape_meta, 
                                    obs_pose_repr=obs_pose_rep,
                                    tx_robot1_robot0=tx_robot1_robot0,
                                    episode_start_pose=episode_start_pose)
                                obs_dict = dict_apply(obs_dict_np, 
                                    lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
                                result = policy.predict_action(obs_dict)
                                raw_action = result['action_pred'][0].detach().to('cpu').numpy()
                                action = get_real_umi_action(raw_action, obs, action_pose_repr)
                                # print('Inference latency:', time.time() - s)
                                
                            # convert policy action to env actions
                            this_target_poses = action
                            assert this_target_poses.shape[1] == len(robots_config) * 7
                            
                            # deal with timing
                            action_timestamps = (np.arange(len(action), dtype=np.float64)
                                ) * dt + obs_timestamps[-1]
                            action_exec_latency = 0.01
                            curr_time = time.time()
                            is_new = action_timestamps > (curr_time + action_exec_latency)
                            if np.sum(is_new) == 0:
                                # exceeded time budget, still do something
                                this_target_poses = this_target_poses[[-1]]
                                # schedule on next available step
                                next_step_idx = int(np.ceil((curr_time - eval_t_start) / dt))
                                action_timestamp = eval_t_start + (next_step_idx) * dt
                                print('Over budget', action_timestamp - curr_time)
                                action_timestamps = np.array([action_timestamp])
                            else:
                                this_target_poses = this_target_poses[is_new]
                                action_timestamps = action_timestamps[is_new]
                                # print('Deal with timing')

                            this_target_poses[:, :3] = this_target_poses[:, :3] * alpha + (1 - alpha) * goal_position
                            this_target_poses[:, 3:6] = this_target_poses[:, 3:6] * alpha + (1 - alpha) * goal_rotation_vector

                            env.exec_actions(
                            actions=this_target_poses,
                            timestamps=action_timestamps,
                            compensate_latency=True
                            )
                            
                            # wait for execution
                            precise_wait(t_cycle_end - frame_latency)
                            iter_idx += steps_per_inference
                            
                        ############## Check For PAUSE
                        press_events = key_counter.get_press_events()
                        for key_stroke in press_events:
                            if key_stroke == KeyCode(char='a'):
                                print("Paused. Press 'a' to resume.")
                                env.robots[0].pause()
                                flag=False
                                time.sleep(5.0)
                                while not flag:
                                    press_events = key_counter.get_press_events()
                                    for key_stroke in press_events:
                                        if key_stroke == KeyCode(char='a'):
                                            flag=True
                                    time.sleep(0.1)

                                # env.robots[0].unpause()
                                print("Resumed.")

                        # print(f"Submitted {len(this_target_poses)} steps of actions.")
                        # # execute actions
                        # env.exec_actions(
                        #     actions=this_target_poses,
                        #     timestamps=action_timestamps,
                        #     compensate_latency=True
                        # )

                        # print(f"Submitted {len(this_target_poses)} steps of actions.")

                        ################# Visualize the body tracking data
                        image = copy.copy(np.array(zed_prediction.image))
                        image_points = copy.copy(np.array(zed_prediction.image_points))
                        if image is not None and image_points is not None:
                            for pt in image_points:
                                image = cv2.circle(image, (int(pt[0]),int(pt[1])), radius=5, color=(255,69,0), thickness=10)
                            cv2.imshow("ZED | 2D View", image)

                        ################ visualize the gopros
                        # episode_id = env.replay_buffer.n_episodes
                        # obs_left_img = obs['camera0_rgb'][-1]
                        # vis_img = np.concatenate([obs_left_img], axis=1)
                        # text = 'Episode: {}, Time: {:.1f}'.format(
                        #     episode_id, time.monotonic() - t_start
                        # )
                        # cv2.putText(
                        #     vis_img,
                        #     text,
                        #     (10,20),
                        #     fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        #     fontScale=0.5,
                        #     thickness=1,
                        #     color=(255,255,255)
                        # )
                        # cv2.imshow('default', vis_img[...,::-1])

                        ################### Plotting            
                        # goal_position_base_links = np.array(goal_position_base_links)
                        # ax.plot3D(goal_position_base_links[-1][:,0], goal_position_base_links[-1][:,1], goal_position_base_links[-1][:,2], 'red')
                        # ax.plot3D([0,current_position[0]], [0,current_position[1]], [0,current_position[2]], 'blue')
                        # plt.pause(0.005)
                        # plt.draw()

                        _ = cv2.pollKey()
                        press_events = key_counter.get_press_events()
                        stop_episode = False
                        for key_stroke in press_events:
                            if key_stroke == KeyCode(char='s'):
                                # Stop episode
                                # Hand control back to human
                                print('Stopped.')
                                stop_episode = True
                        
                        arm_pos_vector.append(arm_position)
                        hand_pos_vector.append(hand_position)
                        pre_hand_pos_vector.append(goal_position)
                        distance_vector.append(distance)
                        duration_vector.append(time.time() - eval_t_start)

                        ########################## Log distance and duration vectors in a file
                        with open(os.path.join(DATA_PATH+DATA_FOLDER, f'arm_pos_log_{COUNTER}.txt'), 'w') as f:
                            for pos in arm_pos_vector:
                                pos_list = pos.tolist()
                                f.write(f"Arm_pos: {pos_list}\n")

                        with open(os.path.join(DATA_PATH+DATA_FOLDER, f'hand_pos_log_{COUNTER}.txt'), 'w') as f:
                            for pos in hand_pos_vector:
                                pos_list = pos.tolist()
                                f.write(f"Hand_pos: {pos_list}\n")
                        
                        with open(os.path.join(DATA_PATH+DATA_FOLDER, f'pre_hand_pos_log_{COUNTER}.txt'), 'w') as f:
                            for pos in pre_hand_pos_vector:
                                pos_list = pos.tolist()
                                f.write(f"Hand_pos: {pos_list}\n")

                        with open(os.path.join(DATA_PATH+DATA_FOLDER, f'distance_log_{COUNTER}.txt'), 'w') as f:
                            f.write(f"Distance: {distance_vector}")

                        with open(os.path.join(DATA_PATH+DATA_FOLDER, f'duration_log_{COUNTER}.txt'), 'w') as f:
                            f.write(f"Duration: {duration_vector}")

                        t_since_start = time.time() - eval_t_start
                        if t_since_start > max_duration:
                            print("Max Duration reached.")
                            stop_episode = True
                        if stop_episode:
                            env.end_episode()
                            break

                except KeyboardInterrupt:
                    print("Interrupted!")
                    # stop robot.
                    env.end_episode()
                
                print("Stopped.")



# %%
if __name__ == '__main__':
    main()