import torch
import torch.nn as nn


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, dropout_rate=0.1):
        super(TransformerEncoder, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        
        # Transformer Encoder Layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=self.num_heads,
            dropout=self.dropout_rate,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """
        src: Tensor of shape (batch_size, seq_len, embed_dim)
        src_mask: None or Tensor for masking in multi-head attention (not used in this example)
        src_key_padding_mask: Tensor of shape (batch_size, seq_len) indicating which elements are padded
        """
        # Applying Transformer Encoder
        output = self.transformer_encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        
        return output