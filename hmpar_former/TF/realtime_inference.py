import torch
import numpy as np
import matplotlib.pyplot as plt
import copy
from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.model_selection import train_test_split
from embedding_layers import SkeletalInputEmbedding
from encoder_layers import TransformerEncoder
from decoder_layers import TransformerDecoder
import TF_helper_functions as hf
import pickle
import numpy as np
import body_tracker.bodytrack_zed as body_tracker
import time
import torch.multiprocessing as mp

from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt


#logfiles
NORMALIZE_DATA_PATH='/home/augustine/lfd_ws/src/HIMPVDP/hmpar_former/data_process/process_data/24_09_09/24_09_09_training_norm.pkl'
MODEL_WEIGHT_PATH = '/home/augustine/lfd_ws/src/HIMPVDP/hmpar_former/models/24_09_09_v1_best_model.pth'

# Create the transformation matrix from the cablibration
transform_matrix_base_to_camera = np.array([[-0.9995, -0.0171, -0.0268, 0.8206],
                                [0.0266, 0.0098, -0.9996, 0.4035],
                                [0.0173, -0.9998, -0.0094, 0.7805],
                                [0, 0, 0, 1]])

WINDOW_SIZE = 15                # Sliding window size for input and output of LSTM model
NUMBER_OF_JOINTS = 6            # Number of joints in the skeleton
WRIST_TO_HAND_LENGHT = 0.15     # Distance from wrist to hand in meters

class Transformer_Model:
    def __init__(self, embed_dim= 128, num_heads= 8, num_layers= 6, num_joints= 6, dropout_rate= 0.1, dof=3, device=None, checkpoint=None):
        self.device = device
        self.checkpoint = checkpoint
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_joints = num_joints
        self.dropout_rate = dropout_rate
        self.dof = dof
        self.input_dim = self.num_joints * self.dof
        
        # Initialize the models with the same configuration as during training
        self.embedding = SkeletalInputEmbedding(self.input_dim).to(self.device)
        self.encoder = TransformerEncoder(self.embed_dim, self.num_heads, self.num_layers, self.dropout_rate).to(self.device)
        self.decoder = TransformerDecoder(self.embed_dim, self.num_heads, self.num_layers, self.num_joints, self.dropout_rate).to(self.device)

        # Load state dicts
        self.embedding.load_state_dict(self.checkpoint['embedding_state_dict'])
        self.encoder.load_state_dict(self.checkpoint['encoder_state_dict'])
        self.decoder.load_state_dict(self.checkpoint['decoder_state_dict'])

        # Set the models to evaluation mode
        self.embedding.eval()
        self.encoder.eval()
        self.decoder.eval()

class Realtime_Inference():
    def __init__(self):
        self.position = torch.tensor([0.0,0.0,0.0], dtype=torch.float32).share_memory_()
        self.predicted_position= torch.tensor([0.0,0.0,0.0], dtype=torch.float32).share_memory_()
        self.image = torch.zeros((1080,1920,4), dtype=torch.uint8).share_memory_()
        self.image_points = torch.zeros((6,2), dtype=torch.float32).share_memory_()

        # Start the multi processing in torch
        mp.set_start_method('spawn', force=True)
        self.p = mp.Process(target=self.inference_update, args=(self.position,self.predicted_position,self.image,self.image_points), daemon=True)
        self.p.start()

    def load_results_from_pickle(self,filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
        
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
    
    def prediction(self,position, velocity, acceleration):
        prediction_inputs = np.array(position)
        prediction_inputs = prediction_inputs.reshape(WINDOW_SIZE, 6, 3)
        prediction_inputs = [hf.reverse_normalization(pos, self.medians_pos, self.iqrs_pos) for pos in prediction_inputs]

        prediction_outputs = None
        predicted_positions  = self.predict(position, velocity, acceleration)
        predicted_positions = np.array(predicted_positions)
        predicted_positions = predicted_positions.reshape(WINDOW_SIZE, 6, 3)
        
  
        prediction_outputs = self.update_base_link(predicted_positions)
        prediction_inputs = self.update_base_link(prediction_inputs)

        prediction_outputs = self.extend_wrist(prediction_outputs,WRIST_TO_HAND_LENGHT)
        prediction_inputs = self.extend_wrist(prediction_inputs,WRIST_TO_HAND_LENGHT)

        return prediction_outputs, prediction_inputs

    def inference_update(self,position,predicted_position,image,image_points):
        fig = plt.figure()
        ax = plt.axes(projection='3d')

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Load the saved model weights
        self.checkpoint = torch.load(MODEL_WEIGHT_PATH, map_location=self.device)

        # Load the saved normalized data
        results = self.load_results_from_pickle(NORMALIZE_DATA_PATH)
        self.medians_pos = results['combined_medians_pos']
        self.iqrs_pos = results['combined_iqrs_pos']
        self.medians_vel = results["combined_medians_vel"]
        self.iqrs_vel = results["combined_iqrs_vel"]
        self.medians_acc = results["combined_medians_acc"]
        self.iqrs_acc = results["combined_iqrs_acc"]

        self.bodytracker = body_tracker.PoseTracker()
        self.model = Transformer_Model(embed_dim= 128, num_heads= 8, num_layers= 6, num_joints= 6, dropout_rate= 0.1, device=self.device, checkpoint=self.checkpoint)
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

            if len(self.data_window) == WINDOW_SIZE:
                # Process the window
                combined_pos = []
                combined_vel = []
                combined_acc = []

                for i in range(0, len(self.data_window)):
                    combined_pos.append(self.data_window[i]['pos'])
                    combined_vel.append(self.data_window[i]['vel'])
                    combined_acc.append(self.data_window[i]['acc'])

                combined_pos = np.array(combined_pos) ### shape (15,6,3)
                combined_vel = np.array(combined_vel) ### shape (15,6,3)
                combined_acc = np.array(combined_acc) ### shape (15,6,3)
                
                # Normalize data
                norm_pos = np.empty_like(combined_pos)
                norm_vel = np.empty_like(combined_vel)
                norm_acc = np.empty_like(combined_acc)
                
                norm_pos = hf.robust_normalize_data_with_clipping(combined_pos, self.medians_pos, self.iqrs_pos, norm_pos)
                norm_vel = hf.robust_normalize_data_with_clipping(combined_vel, self.medians_vel, self.iqrs_vel, norm_vel)
                norm_acc = hf.robust_normalize_data_with_clipping(combined_acc, self.medians_acc, self.iqrs_acc, norm_acc)

                norm_pos = np.array([norm_pos]) ### shape (1,15,6,3)
                norm_vel = np.array([norm_vel]) ### shape (1,15,6,3)
                norm_acc = np.array([norm_acc]) ### shape (1,15,6,3)

                # Do the prediction
                predicted_positions_base_link,positions_base_link = self.prediction(norm_pos, norm_vel, norm_acc)
                position_new_values = torch.tensor(positions_base_link[-1][0,0:3], dtype=torch.float32)
                predicted_new_values = torch.tensor(predicted_positions_base_link[-1][0,0:3], dtype=torch.float32)

                position.copy_(position_new_values)
                predicted_position.copy_(predicted_new_values)
                image.copy_(torch.tensor(self.bodytracker.image.get_data(), dtype=torch.uint8).to(self.model.device))
                image_points.copy_(torch.tensor(self.bodytracker.image_points, dtype=torch.float32).to(self.model.device))
             
                # PLOTTTING
                # plt.cla()
                # ax.axes.set_xlim3d(left=0, right=1.5) 
                # ax.axes.set_ylim3d(bottom=-1, top=0) 
                # ax.axes.set_zlim3d(bottom=0.0, top=1.0) 

                # ax.plot3D(positions_base_link[-1][:,0], positions_base_link[-1][:,1], positions_base_link[-1][:,2], 'blue')
                # ax.plot3D(predicted_positions_base_link[0][:,0], predicted_positions_base_link[0][:,1], predicted_positions_base_link[0][:,2], 'red')
                # plt.pause(0.005)

    def predict(self,combined_X_pos,combined_X_vel,combined_X_acc):
        # Convert to PyTorch tensors
        X_pos_tensor = torch.tensor(combined_X_pos, dtype=torch.float32).to(self.device)
        X_vel_tensor = torch.tensor(combined_X_vel, dtype=torch.float32).to(self.device)
        X_acc_tensor = torch.tensor(combined_X_acc, dtype=torch.float32).to(self.device)

        predicted_positions = []

        # Encoder pass
        inputembeddings = self.model.embedding(X_pos_tensor, X_vel_tensor)
        memory = self.model.encoder(inputembeddings, src_key_padding_mask=None)

        # Initialize the start token for decoding
        current_pos = X_pos_tensor[:, -1:, :, :]
        current_vel= X_vel_tensor[:, -1:, :, :]

        for i in range(WINDOW_SIZE):
            # Embed the current position and velocity
            current_embeddings = self.model.embedding(current_pos, current_vel)
            
            # Decoder pass
            output = self.model.decoder(current_embeddings, memory, tgt_key_padding_mask=None, memory_key_padding_mask=None)
    
            # Update current_pos for the next prediction
            old_pos= current_pos
            current_pos = output[:, :, :, :].detach()  # only take the last timestep

            # Get output
            predicted_positions.append(current_pos.squeeze().cpu().numpy())
            output = output.where(~torch.isnan(output), torch.zeros_like(output))
        
            ############# UPDATE VELOCITY WITH NORMALIZATION
            # Convert to numpy
            numpy_current_pos = current_pos.cpu().numpy() 
            numpy_old_pos = old_pos.cpu().numpy()
            
            # Denormalize to do the substraction 
            denor_current_pos = hf.reverse_normalization(numpy_current_pos, self.medians_pos, self.iqrs_pos)
            denor_old_pos = hf.reverse_normalization(numpy_old_pos, self.medians_pos, self.iqrs_pos)
            denor_current_pos = denor_current_pos.reshape(1, 6, 3)
            denor_old_pos = denor_old_pos.reshape(1, 6, 3)
            
            # Get the velocity
            denor_current_vel = (denor_current_pos-denor_old_pos)/0.1
            
            # Normalize the velocity
            numpy_current_vel = np.empty_like(denor_current_vel)
            numpy_current_vel = hf.robust_normalize_data_with_clipping(denor_current_vel, self.medians_vel, self.iqrs_vel, numpy_current_vel)
            current_vel = torch.tensor(numpy_current_vel, dtype=torch.float32).to(self.device)
        
        normalized_predicted_positions = [hf.reverse_normalization(pos, self.medians_pos, self.iqrs_pos) for pos in predicted_positions]
        return normalized_predicted_positions
    
def main():
    realtime_inference = Realtime_Inference()   
    while True:
        time.sleep(1) 
    
if __name__ == "__main__":
    main()
