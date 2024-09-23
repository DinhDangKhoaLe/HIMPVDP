
import cv2
import sys
import time
import pyzed.sl as sl
import numpy as np
import argparse
import copy
import pickle
from threading import Thread, Event

UPDATE_RATE = 10
PICKLE_FILE_FOLDER = '/home/augustine/lfd_ws/src/skill_transfer/hmpar_former/data_process/raw_data/24_09_14/'
PICKLE_FILE_NAME = 'Khoa_11.pkl'
PICKLE_FILE_PATH = PICKLE_FILE_FOLDER + PICKLE_FILE_NAME

class PoseTracker:
    def __init__(self):
        print('Init Zed Pose Tracker')
        self.stop_event = Event()
        self.camera_setup()

    def camera_setup(self):
        # Create a Camera object
        self.zed = sl.Camera()

        # Create a InitParameters object and set configuration parameters
        self.init_params = sl.InitParameters()
        self.init_params.camera_resolution = sl.RESOLUTION.HD1080  # Use HD1080 video mode
        self.init_params.coordinate_units = sl.UNIT.METER          # Set coordinate units
        self.init_params.depth_mode = sl.DEPTH_MODE.ULTRA
        self.init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP

        # Open the camera
        err = self.zed.open(self.init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            exit(1)

        print('Init Zed Camera')
        # Get ZED camera information
        camera_info = self.zed.get_camera_information()
        self.display_resolution = sl.Resolution(min(camera_info.camera_configuration.resolution.width, 1920), min(camera_info.camera_configuration.resolution.height, 1080))
        self.image_scale = [self.display_resolution.width / camera_info.camera_configuration.resolution.width
                 , self.display_resolution.height / camera_info.camera_configuration.resolution.height]
        # print("DISPLAY RESOLUTION: ", display_resolution.width, "x", display_resolution.height)

        # Body parameters
        self.body_param = sl.BodyTrackingParameters()
        self.body_param.enable_tracking = True                # Track people across images flow
        self.body_param.enable_body_fitting = True            # Smooth skeleton move
        self.body_param.detection_model = sl.BODY_TRACKING_MODEL.HUMAN_BODY_ACCURATE # Can be FAST OR MEDIUM OR ACCURATE
        self.body_param.body_format = sl.BODY_FORMAT.BODY_18  # Choose the BODY_FORMAT you wish to use (BODY_18, BODY_34, BODY_38)

        # Enable Positional tracking (mandatory for object detection)
        positional_tracking_parameters = sl.PositionalTrackingParameters()
        positional_tracking_parameters.set_as_static = True # For static camera
        self.zed.enable_positional_tracking(positional_tracking_parameters)
        
        # Enable Body Detection module
        self.zed.enable_body_tracking(self.body_param)

        # Enable the runtime parameters for the body tracking
        self.body_runtime_param = sl.BodyTrackingRuntimeParameters()
        self.body_runtime_param.detection_confidence_threshold = 80

        # Enable the runtime parameters for 3D position
        # runtime_3D_params = sl.RuntimeParameters()
        # runtime_3D_params.measure3D_reference_frame = sl.REFERENCE_FRAME.WORLD

        # Zed object and image
        self.bodies = sl.Bodies()
        self.image = sl.Mat()
        self.key_wait = 10 

        # Time for calculating velocity and acceleration
        self.t1 = time.time()  
        self.t2 = time.time()  
        # World points
        self.time_stamp = []
        self.image_points =  None
        self.world_points =  None 
        self.world_points_vel = None
        self.world_points_acc = None 
        
        self.dataset = {
            'timestamp': np.zeros((0, 1),dtype=np.float64),
            'points': np.zeros((0, 6, 3), dtype=np.float32),
            'velocity': np.zeros((0, 6, 3), dtype=np.float32),
            'acceleration': np.zeros((0, 6, 3), dtype=np.float32)
        }

        # Start the update camera thread
        self.camera_thread = Thread(target=self.camera_update,args= (1.0/UPDATE_RATE,))
        self.camera_thread.start()

    def camera_update(self, time_seconds):
        world_points_prev =  np.zeros((6, 3), dtype=np.float32)
        world_points_prev_vel =  np.zeros((6, 3), dtype=np.float32)
        try:
            while not self.stop_event.is_set():
                # Start the time measurement
                self.t1 = time.time()

                # Grab an image
                if self.zed.grab() == sl.ERROR_CODE.SUCCESS:
                    # Retrieve left image
                    self.zed.retrieve_image(self.image, sl.VIEW.LEFT, sl.MEM.CPU, self.display_resolution)
                    # Retrieve bodies
                    self.zed.retrieve_bodies(self.bodies, self.body_runtime_param)
                    if len(self.bodies.body_list) > 0:
                        for body in self.bodies.body_list:
                            if body.confidence > 90.0:
                                ############ For 34 points model
                                # self.image_points = np.float32([body.keypoint_2d[16],body.keypoint_2d[13],body.keypoint_2d[12],body.keypoint_2d[5],body.keypoint_2d[6],body.keypoint_2d[9]])
                                # self.world_points = np.float32([body.keypoint[16],body.keypoint[13],body.keypoint[12],body.keypoint[5],body.keypoint[6],body.keypoint[9]])
                            
                                ############# For 18 points model
                                self.image_points = np.float32([body.keypoint_2d[4],body.keypoint_2d[3],body.keypoint_2d[2],body.keypoint_2d[5],body.keypoint_2d[6],body.keypoint_2d[7]])
                                self.world_points = np.float32([body.keypoint[4],body.keypoint[3],body.keypoint[2],body.keypoint[5],body.keypoint[6],body.keypoint[7]])
                        
                        # Calulate velocity and acceleration
                        delta_t=self.t1-self.t2
                        if world_points_prev is not None and self.world_points is not None:
                            if delta_t > 0 and delta_t < 1:
                                # calculate velocity
                                self.world_points_vel=np.float32((self.world_points-world_points_prev)/delta_t) 
                                # calculate acceleration
                                if world_points_prev_vel is not None:
                                    self.world_points_acc=np.float32((self.world_points_vel-world_points_prev_vel)/delta_t)           
                                world_points_prev_vel = self.world_points_vel               
                            else:
                                world_points_prev = [] 
                                world_points_prev_vel = []
                                self.world_points_acc = []
                        world_points_prev = self.world_points
                        
                        # Append data to dataset
                        self.dataset['timestamp'] = np.concatenate((self.dataset['timestamp'], np.float64(time.time()).reshape(1,1)), axis=0)
                        self.dataset['points'] = np.concatenate((self.dataset['points'], self.world_points.reshape(1,6,3)), axis=0)      
                        self.dataset['velocity'] = np.concatenate((self.dataset['velocity'], self.world_points_vel.reshape(1,6,3)), axis=0)
                        self.dataset['acceleration'] = np.concatenate((self.dataset['acceleration'], self.world_points_acc.reshape(1,6,3)), axis=0)

                        # Update 2D view
                        # image_left_ocv = self.image.get_data()
                        # for pt in self.image_points:
                        #     image_left_ocv = cv2.circle(image_left_ocv, (int(pt[0]),int(pt[1])), radius=5, color=(255,69,0), thickness=10)
                        # cv2.imshow("ZED | 2D View", image_left_ocv)

                        # # Q for quit
                        # key = cv2.waitKey(self.key_wait)
                        # if key == 113: # for 'q' key
                        #     print("Exiting...")
                        #     break

                    if self.t1-self.t2 != 0:
                        FPS=1/(self.t1-self.t2)
                        # print(f"FPS: {int(FPS)}")  # Print the FPS
                    self.t2=copy.copy(self.t1)
                
                time.sleep(time_seconds - time.time() % time_seconds)
            
            # Clean up and dump data to pickle file
            self.cleanup()

        except KeyboardInterrupt:
            self.stop_event.set()
            print("Key board interupt...")

    def write_data_to_pickle(self):
        with open(PICKLE_FILE_PATH, 'wb') as f:
            pickle.dump(self.dataset, f)
        print(f"Data has been written to {PICKLE_FILE_PATH}")
    
    def cleanup(self):
        print("Entering cleanup...")
        self.image.free(sl.MEM.CPU)
        self.zed.disable_body_tracking()
        self.zed.disable_positional_tracking()
        self.zed.close()
        cv2.destroyAllWindows()
        # self.write_data_to_pickle()

def main():
    pose_tracker = PoseTracker()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_svo_file', type=str, help='Path to an .svo file, if you want to replay it',default = '')
    parser.add_argument('--ip_address', type=str, help='IP Adress, in format a.b.c.d:port or a.b.c.d, if you have a streaming setup', default = '')
    parser.add_argument('--resolution', type=str, help='Resolution, can be either HD2K, HD1200, HD1080, HD720, SVGA or VGA', default = '')
    opt = parser.parse_args()
    if len(opt.input_svo_file)>0 and len(opt.ip_address)>0:
        print("Specify only input_svo_file or ip_address, or none to use wired camera, not both. Exit program")
        exit()
    main() 