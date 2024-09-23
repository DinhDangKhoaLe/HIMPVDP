import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)
os.chdir(ROOT_DIR)

import json
import pathlib
import click
import zarr
import pickle
import numpy as np
import cv2
import av
import multiprocessing
import concurrent.futures
from tqdm import tqdm
from collections import defaultdict
import cv2
import h5py
import shutil
from umi.common.cv_util import (
    parse_fisheye_intrinsics,
    FisheyeRectConverter,
    get_image_transform, 
    draw_predefined_mask,
    inpaint_tag,
)

# import numpy as np
# import roboticstoolbox as rtb
# from spatialmath import SE3
        
def process_episode(plan_episode, fisheye_converter, out_res, demos_path, dataset_dir, 
                    mirror_swap, no_mirror, episode_idx):
    data_dict = {
        '/observations/images/camera0_rgb': [],
        '/observations/qpos': [],
        '/observations/qvel': [],
        '/observations/effort': [],
        '/action': [],
    }
    
    # ur10 = rtb.models.DH.UR10()
    # initial_guess_deg = [-37.26, -119.02,-122.68, 57.14, 88.44, -0.46]
    # initial_guess_rad = np.deg2rad(initial_guess_deg)
    
    # mirror swap  
    is_mirror = None
    if mirror_swap:
        ow, oh = out_res
        mirror_mask = np.ones((oh, ow, 3), dtype=np.uint8)
        mirror_mask = draw_predefined_mask(
            mirror_mask, color=(0, 0, 0), mirror=True, gripper=False, finger=False)
        is_mirror = (mirror_mask[..., 0] == 0)
    
    # Add gripper data to data_dict
    grippers = plan_episode['grippers']
    for gripper_id, gripper in enumerate(grippers):    
        eef_pose = gripper['tcp_pose']
        # demo_start_pose = np.empty_like(eef_pose)
        # demo_start_pose[:] = gripper['demo_start_pose']
        # print("Demo start pose:", demo_start_pose)
        # joint_configs = []
        # joint_guess = None
        # # Convert end effector pose to joint configuration
        # for pose in eef_pose:
        #     # Convert pose to transformation matrix
        #     t = pose[:3]
        #     rpy = pose[3:]
        #     R = SE3.Rx(rpy[0]) * SE3.Ry(rpy[1]) * SE3.Rz(rpy[2])
        #     T = SE3(t) * R
        #     print(pose)
        #     # Define joint guess
        #     if joint_guess is None:
        #         joint_guess = initial_guess_rad
        #     else:
        #         joint_guess = joint_configs[-1]
                
        #     # Solve inverse kinematics
        #     joint_config = ur10.ikine_LM(T,q0 = initial_guess_rad)
        #     if joint_config.success == True:
        #         joint_configs.append(joint_config.q)
        #     # else:
        #         # print("Inverse kinematics failed to converge. No solution found.")
        # Add gripper width and joint config to action     
        gripper_widths = gripper['gripper_width']
        actions = np.column_stack((eef_pose, gripper_widths)) # (n_steps, 7)
        for action in actions:
            data_dict['/observations/qpos'].append(action.astype(np.float32))
            data_dict['/observations/qvel'].append(np.zeros_like(action, dtype=np.float32))
            data_dict['/observations/effort'].append(np.zeros_like(action, dtype=np.float32))
            data_dict['/action'].append(action.astype(np.float32))
    
    # Add camera data to data_dict
    cameras = plan_episode['cameras']
    for cam_id, camera in enumerate(cameras):
        video_path_rel = camera['video_path']
        video_path = demos_path.joinpath(video_path_rel).absolute()
        assert video_path.is_file()
        pkl_path = os.path.join(os.path.dirname(video_path), 'tag_detection.pkl')
        tag_detection_results = pickle.load(open(pkl_path, 'rb'))
        dataset_name = f'episode_{episode_idx}'
        dataset_path = os.path.join(dataset_dir, dataset_name)
        video_start_frame, video_end_frame = camera['video_start_end']

        # convert video_path to string
        with av.open(str(video_path)) as container:
            in_stream = container.streams.video[0]
            ih, iw = in_stream.height, in_stream.width
            # in_stream.thread_type = "AUTO"
            in_stream.thread_count = 1
            resize_tf = get_image_transform(
                in_res=(iw, ih),
                out_res=out_res
            )
            for frame_idx, frame in tqdm(enumerate(container.decode(in_stream)), total=in_stream.frames, leave=False):
                if frame_idx < video_start_frame:
                    continue
                if frame_idx > video_end_frame-1:
                    continue
                img = frame.to_ndarray(format='rgb24')
                
                # inpaint tags
                this_det = tag_detection_results[frame_idx]
                all_corners = [x['corners'] for x in this_det['tag_dict'].values()]
                for corners in all_corners:
                    img = inpaint_tag(img, corners)
    
                # mask out gripper
                img = draw_predefined_mask(img, color=(0, 0, 0), 
                    mirror=no_mirror, gripper=True, finger=False)
                # resize
                if fisheye_converter is None:
                    img = resize_tf(img)
                else:
                    img = fisheye_converter.forward(img)
                    
                # handle mirror swap
                if mirror_swap:
                    img[is_mirror] = img[:, ::-1, :][is_mirror]
                
                data_dict['/observations/images/camera0_rgb'].append(img)   
    
    with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024**2*2) as root:
        oh, ow = out_res
        root.attrs['sim'] = False
        root.attrs['compress'] = False
        obs = root.create_group('observations')
        image = obs.create_group('images')
        cam_name = 'camera0_rgb'
        max_timesteps = len(data_dict['/observations/images/camera0_rgb'])
        _ = image.create_dataset(cam_name, (max_timesteps, ow, oh, 3), dtype='uint8',
                                    chunks=(1, ow, oh, 3), )
        _ = obs.create_dataset('qpos', (max_timesteps, 7))
        _ = obs.create_dataset('qvel', (max_timesteps, 7))
        _ = obs.create_dataset('effort', (max_timesteps, 7))
        _ = root.create_dataset('action', (max_timesteps, 7))

        for name, array in data_dict.items():
            root[name][...] = array

@click.command()
@click.argument('input', nargs=-1)
@click.option('-o', '--output', required=True, help='Zarr path')
@click.option('-or', '--out_res', type=str, default='224,224')
@click.option('-of', '--out_fov', type=float, default=None)
@click.option('-cl', '--compression_level', type=int, default=99)
@click.option('-nm', '--no_mirror', is_flag=True, default=False, help="Disable mirror observation by masking them out")
@click.option('-ms', '--mirror_swap', is_flag=True, default=False)
@click.option('-n', '--num_workers', type=int, default=None)
def main(input, output, out_res, out_fov, compression_level, 
         no_mirror, mirror_swap, num_workers):
    if os.path.isdir(output):
        if click.confirm(f'Output directory {output} exists! Overwrite?', abort=True):
            shutil.rmtree(output)
            os.makedirs(output)
        
    out_res = tuple(int(x) for x in out_res.split(','))

    if num_workers is None:
        num_workers = multiprocessing.cpu_count()
    cv2.setNumThreads(1)
            
    fisheye_converter = None
    if out_fov is not None:
        intr_path = pathlib.Path(os.path.expanduser(ipath)).absolute().joinpath(
            'calibration',
            'gopro_intrinsics_2_7k.json'
        )
        opencv_intr_dict = parse_fisheye_intrinsics(json.load(intr_path.open('r')))
        fisheye_converter = FisheyeRectConverter(
            **opencv_intr_dict,
            out_size=out_res,
            out_fov=out_fov
        )
    
    # dump lowdim data to replay buffer
    # generate argument for videos
    n_grippers = None
    n_cameras = None
    episode_idx = 0

    tasks = []
    for ipath in input:
        ipath = pathlib.Path(os.path.expanduser(ipath)).absolute()
        demos_path = ipath.joinpath('demos')
        dataset_dir = ipath.joinpath('ACT_dataset')
        if not os.path.isdir(dataset_dir):
            os.makedirs(dataset_dir)
            
        plan_path = ipath.joinpath('dataset_plan.pkl')
        if not plan_path.is_file():
            print(f"Skipping {ipath.name}: no dataset_plan.pkl")
            continue

        plan = pickle.load(plan_path.open('rb'))

        for plan_episode in plan:
            # check if all episodes have the same number of grippers and cameras
            grippers = plan_episode['grippers']
            if n_grippers is None:
                n_grippers = len(grippers)
            else:
                assert n_grippers == len(grippers) 
            cameras = plan_episode['cameras']
            if n_cameras is None:
                n_cameras = len(cameras)
            else:
                assert n_cameras == len(cameras)

            tasks.append((plan_episode, fisheye_converter, out_res, demos_path, dataset_dir, 
                          mirror_swap, no_mirror, episode_idx))
            episode_idx += 1

    with tqdm(total=len(plan)) as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = set()
            if len(futures) >= num_workers:
                    # limit number of inflight tasks
                    completed, futures = concurrent.futures.wait(futures, 
                        return_when=concurrent.futures.FIRST_COMPLETED)
                    pbar.update(len(completed))
            for task in tasks:
                future = executor.submit(process_episode, *task)
                futures.add(future)
            completed, futures = concurrent.futures.wait(futures)
            pbar.update(len(completed))

    print(f"Done! {len(plan)} videos used in total!")
if __name__ == "__main__":
    main()
