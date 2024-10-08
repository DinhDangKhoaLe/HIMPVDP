import torch.nn as nn

from models.decoder import VelDecoder
from models.encoder import Encoder


class DEGlobalPosetrack(nn.Module):
    def __init__(self, args):
        super(DEGlobalPosetrack, self).__init__()
        self.args = args
        self.pose_encoder = Encoder(args=self.args, input_size=2)
        self.vel_encoder = Encoder(args=self.args, input_size=2)
        self.vel_decoder = VelDecoder(args=self.args, out_features=2, input_size=2)

    def forward(self, pose=None, vel=None):
        outputs = []

        (hidden_vel, cell_vel) = self.vel_encoder(vel.permute(1, 0, 2))
        hidden_vel = hidden_vel.squeeze(0)
        cell_vel = cell_vel.squeeze(0)

        (hidden_pose, cell_pose) = self.pose_encoder(pose.permute(1, 0, 2))
        hidden_pose = hidden_pose.squeeze(0)
        cell_pose = cell_pose.squeeze(0)

        VelDec_inp = vel[:, -1, :]

        hidden_dec = hidden_pose + hidden_vel
        cell_dec = cell_pose + cell_vel
        return self.vel_decoder(VelDec_inp, hidden_dec, cell_dec)
