import torch
import torch.nn as nn

class TransformerDecoder(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, num_joints, dropout_rate=0.1):
        super(TransformerDecoder, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.num_joints = num_joints

        # Transformer Decoder Layer
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.embed_dim,  # Keeping d_model consistent with embed_dim
            nhead=self.num_heads,
            dropout=self.dropout_rate,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=self.num_layers)

        # Output layer to convert decoder output to joint position dimension
        self.output_layer = nn.Linear(self.embed_dim, self.num_joints * 3)  # Assuming output per joint is a 3D position

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        """
        tgt: Tensor of shape (batch_size, output_seq_len, embed_dim), initially could be start token or zero vectors
        memory: Tensor of shape (batch_size, input_seq_len, embed_dim), output from the Transformer encoder
        tgt_mask: Mask to ensure the decoder's predictions are based only on past positions
        memory_mask: Optional, to mask encoder outputs if necessary
        tgt_key_padding_mask: Tensor of shape (batch_size, output_seq_len) for masking target sequences
        memory_key_padding_mask: Tensor of shape (batch_size, input_seq_len) for masking memory sequences
        """

        # Transformer Decoder
        output = self.transformer_decoder(
            tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask
        )

        # Project to joint position dimensions
        output = self.output_layer(output)

        # Reshape to (batch_size, seq_len, num_joints, 3)
        output = output.view(output.size(0), output.size(1), self.num_joints, 3)

        return output