import torch
import torch.nn as nn
import numpy as np

class SkeletalInputEmbedding(nn.Module):
    def __init__(self, input_dim, embed_dim=64, device='cuda'):
        super(SkeletalInputEmbedding, self).__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.device = device

        # Linear layer to project input_dim to embed_dim
        # Linear layers to project input_dim to embed_dim for each type of data
        self.position_embed = nn.Linear(input_dim, embed_dim)
        self.velocity_embed = nn.Linear(input_dim, embed_dim)
        #self.acceleration_embed = nn.Linear(input_dim, embed_dim)

    def forward(self, joint_positions, joint_velocities):
        # joint_positions shape: (batch_size, seq_len, num_joints, dof)
        batch_size, seq_len, num_joints, dof = joint_positions.size()
        input_dim = num_joints * dof  # Total input dimension

        # Reshape to (batch_size * seq_len, num_joints * dof)
        joint_positions = joint_positions.view(batch_size * seq_len, input_dim)
        joint_velocities = joint_velocities.view(batch_size * seq_len, input_dim)
        #joint_accelerations = joint_accelerations.view(batch_size * seq_len, input_dim)

        # Apply the linear layers to project to embed_dim
        position_embeddings = self.position_embed(joint_positions)
        velocity_embeddings = self.velocity_embed(joint_velocities)
        #acceleration_embeddings = self.acceleration_embed(joint_accelerations)

        # Concatenate embeddings along the last dimension
        combined_embeddings = torch.cat((position_embeddings, velocity_embeddings), dim=-1)

        # Reshape back to (batch_size, seq_len, embed_dim * 3)
        combined_embeddings = combined_embeddings.view(batch_size, seq_len, self.embed_dim * 2)

        # Calculate positional encoding
        positional_encoding = self.get_sinusoidal_encoding(seq_len, self.embed_dim * 2).to(self.device)
        positional_encoding = positional_encoding.unsqueeze(0).expand(batch_size, seq_len, self.embed_dim * 2)

        # Add positional encoding to the embeddings
        combined_embeddings += positional_encoding

        return combined_embeddings


    def get_sinusoidal_encoding(self, total_len, embed_dim):
        position = torch.arange(0, total_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * -(np.log(10000.0) / embed_dim))
        
        pe = torch.zeros(total_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe