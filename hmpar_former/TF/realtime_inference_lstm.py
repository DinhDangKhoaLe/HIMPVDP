import torch
import numpy as np
import copy
import body_tracker.bodytrack_zed as body_tracker
import time
import torch.multiprocessing as mp
import matplotlib.pyplot as plt
from utils.others import load_model, speed2pos3d # for LSTM model
import argparse

# LSTM model path
MODEL_WEIGHT_PATH = '/home/augustine/lfd_ws/src/skill_transfer/hmpar_former/models/Khoa_epoch199.pth'

# Create the transformation matrix from the cablibration
transform_matrix_base_to_camera = np.array([[-0.9995, -0.0171, -0.0268, 0.8206],
                                [0.0266, 0.0098, -0.9996, 0.4035],
                                [0.0173, -0.9998, -0.0094, 0.7805],
                                [0, 0, 0, 1]])

WINDOW_SIZE = 15                # Sliding window size for input and output of LSTM model
NUMBER_OF_JOINTS = 6            # Number of joints in the skeleton
WRIST_TO_HAND_LENGHT = 0.15     # Distance from wrist to hand in meters

class LSTM_Model:
    def __init__(self):
        self.opt = self.parse_option()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint = torch.load(MODEL_WEIGHT_PATH, map_location=self.device)
        self.model = load_model(self.opt,18,18) # 18 is 6*3 (6 joints, 3 coordinates)
    
    def predict(self,norm_pos):
        self.model.eval()
        obs_pose = torch.tensor(norm_pos, dtype=torch.float32).to(self.opt.device)
        obs_speed = (obs_pose[1:] - obs_pose[:-1])
        obs_pose = obs_pose.unsqueeze(0)
        obs_speed = obs_speed.unsqueeze(0)
        with torch.no_grad():
            (speed_preds,speed_preds2) = self.model(pose=obs_pose, vel=obs_speed)
            preds_p = speed2pos3d(speed_preds, obs_pose)
            pred_pose_np = preds_p.squeeze().cpu().numpy()
            return pred_pose_np
        
    def parse_option(self):
        #initialize the parser
        opt = argparse.Namespace(
            device='cuda',
            input=WINDOW_SIZE,
            output=WINDOW_SIZE,
            hidden_size=1000,
            batch_size=64,
            hardtanh_limit=100,
            load_ckpt=MODEL_WEIGHT_PATH,
            n_layers=1,
            dropout_encoder=0,
            dropout_pose_decoder=0,
            dropout_mask_decoder=0,
            test_output=False,
            dataset_name=''
        )
        opt.stride = opt.input
        opt.skip = 1
        opt.loader_shuffle = True
        opt.pin_memory = False
        opt.model_name = 'lstm_vel'
        return opt

class Realtime_Inference():
    #constructor
    def __init__(self,realtime=True):
        self.model = LSTM_Model()
        self.position = torch.tensor([0.0,0.0,0.0], dtype=torch.float32).share_memory_()
        self.predicted_position= torch.tensor([0.0,0.0,0.0], dtype=torch.float32).share_memory_()
        self.image = torch.zeros((1080,1920,4), dtype=torch.uint8).share_memory_()
        self.image_points = torch.zeros((6,2), dtype=torch.float32).share_memory_()

        # Start the multi processing in torch
        if realtime:
            mp.set_start_method('spawn', force=True)
            self.p = mp.Process(target=self.inference_update, args=(self.position,self.predicted_position,self.image,self.image_points), daemon=True)
            self.p.start()

    def update_base_link(self,position):
        # Update to baselink for position
        predicted_positions_base_link_outputs = []
        for j in range(0, len(position)):
            predicted_positions_base_link = []
            for i in range(0, len(position[0])):
                joint_predict_pos = copy.copy(position[j][i,:])
                joint_predict_pos[1]*=-1 # flip to realsense frame, z is poistive forward
                joint_predict_pos[2]*=-1
                joint_predict_pos = np.append(joint_predict_pos,1)
                predicted_positions_base_link.append(np.matmul(transform_matrix_base_to_camera,np.array(joint_predict_pos).transpose()))
            predicted_positions_base_link = np.array(predicted_positions_base_link)
            predicted_positions_base_link_outputs.append(predicted_positions_base_link)
            
        predicted_positions_base_link_outputs = np.array(predicted_positions_base_link_outputs)
        return predicted_positions_base_link_outputs

    def extend_wrist(self,position,extended_distance=1.0):
        # Extend the wrist to the hand because the hand prediction from zed is not accurate
        for i in range(0, len(position)):
            wrist_pose = position[i][0,0:3]
            elbow_pose = position[i][1,0:3]
            vector = wrist_pose - elbow_pose  
            extended_vector = wrist_pose + (vector / np.linalg.norm(vector)) * extended_distance
            position[i][0,0:3] = extended_vector
        return position

    def prediction(self,position):
        prediction_inputs = np.array(position)
        prediction_outputs = None
        
        predicted_positions  = self.model.predict(position)
        predicted_positions = predicted_positions.reshape(WINDOW_SIZE, 6, 3)
        prediction_inputs = prediction_inputs.reshape(WINDOW_SIZE, 6, 3)
  
        prediction_outputs = self.update_base_link(predicted_positions)
        prediction_inputs = self.update_base_link(prediction_inputs)

        prediction_outputs = self.extend_wrist(prediction_outputs,WRIST_TO_HAND_LENGHT)
        prediction_inputs = self.extend_wrist(prediction_inputs,WRIST_TO_HAND_LENGHT)

        return prediction_outputs, prediction_inputs
    
    
    def inference_update(self,position,predicted_position,image,image_points):
        ######### PLOTTING
        fig = plt.figure()
        ax = plt.axes(projection='3d')

        self.bodytracker = body_tracker.PoseTracker()
        self.data_window = []

        while True:
            # Check the body tracker is having data or not
            if self.bodytracker.world_points is None or self.bodytracker.world_points_vel is None or self.bodytracker.world_points_acc is None:
                continue

            # Pop the first element of the window
            if len(self.data_window) == WINDOW_SIZE:
                self.data_window.pop(0)
            
            # Get data from BodyTracker and add to the window
            if len(self.bodytracker.world_points) != 0 and len(self.bodytracker.world_points_vel) != 0 and len(self.bodytracker.world_points_acc) != 0:     
                self.data_window.append({'pos':self.bodytracker.world_points,'vel':self.bodytracker.world_points_vel,'acc':self.bodytracker.world_points_acc})

            # Process the window
            if len(self.data_window) == WINDOW_SIZE:
                combined_pos = []
                combined_vel = []
                combined_acc = []

                for i in range(0, len(self.data_window)):
                    combined_pos.append(self.data_window[i]['pos'])
                    combined_vel.append(self.data_window[i]['vel'])
                    combined_acc.append(self.data_window[i]['acc'])

                combined_pos = np.array(combined_pos) ### shape (WINDOW_SIZE,6,3)
                combined_vel = np.array(combined_vel) ### shape (WINDOW_SIZE,6,3)
                combined_acc = np.array(combined_acc) ### shape (WINDOW_SIZE,6,3)

                norm_pos = np.array(combined_pos) ### shape (1,WINDOW_SIZE,6,3)
                norm_vel = np.array(combined_vel) ### shape (1,WINDOW_SIZE,6,3)
                norm_acc = np.array(combined_acc) ### shape (1,WINDOW_SIZE,6,3)

                norm_pos = norm_pos.reshape(1, WINDOW_SIZE, 18) ### shape (1,WINDOW_SIZE,18)
                norm_vel = norm_vel.reshape(1, WINDOW_SIZE, 18) ### shape (1,WINDOW_SIZE,18)
                norm_acc = norm_acc.reshape(1, WINDOW_SIZE, 18) ### shape (1,WINDOW_SIZE,18)

                ######### do the prediction
                predicted_positions_base_link,positions_base_link = self.prediction(norm_pos[0])
                position_new_values = torch.tensor(positions_base_link[-1][0,0:3], dtype=torch.float32)
                predicted_new_values = torch.tensor(predicted_positions_base_link[-1][0,0:3], dtype=torch.float32)

                position.copy_(position_new_values)
                predicted_position.copy_(predicted_new_values)
                image.copy_(torch.tensor(self.bodytracker.image.get_data(), dtype=torch.uint8).to(self.model.opt.device))
                image_points.copy_(torch.tensor(self.bodytracker.image_points, dtype=torch.float32).to(self.model.opt.device))

                # PLOTTTING
                plt.cla()
                ax.axes.set_xlim3d(left=0, right=1.5) 
                ax.axes.set_ylim3d(bottom=-1, top=0) 
                ax.axes.set_zlim3d(bottom=0.0, top=1.0) 

                ax.plot3D(positions_base_link[-1][:,0], positions_base_link[-1][:,1], positions_base_link[-1][:,2], 'blue')
                ax.plot3D(predicted_positions_base_link[-1][:,0], predicted_positions_base_link[-1][:,1], predicted_positions_base_link[-1][:,2], 'red')
                plt.pause(0.005)

    def close(self):
        if self.p.is_alive():
            self.p.terminate()
        self.p.join()

def main():
    realtime_inference = Realtime_Inference()    
    while True:
        time.sleep(1)
if __name__ == "__main__":
    main()
