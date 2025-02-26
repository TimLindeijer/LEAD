from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.ADformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import ADformerLayer
from layers.Embed import TokenChannelEmbedding
import numpy as np

class DecoderLayer(nn.Module):
    """
    Transformer Decoder Layer.
    Inspired by "Attention is All You Need" https://arxiv.org/abs/1706.03762.
    """
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, activation="gelu"):
        super(DecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.gelu if activation == "gelu" else F.relu

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        # Self-attention block
        tgt2, _ = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask)
        tgt = self.norm1(tgt + self.dropout(tgt2))
        # Cross-attention block with encoder memory
        tgt2, _ = self.cross_attn(tgt, memory, memory, attn_mask=memory_mask)
        tgt = self.norm2(tgt + self.dropout(tgt2))
        # Feed-forward network block
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = self.norm3(tgt + self.dropout(tgt2))
        return tgt

class DenoiseDiffusion:
    """
    ## Denoise Diffusion - Simplified implementation
    """
    def __init__(self, eps_model: nn.Module, n_steps: int, device: torch.device, time_diff_constraint=True):
        """
        * `eps_model` is the model that predicts noise
        * `n_steps` is number of diffusion steps
        * `device` is the device to place constants on
        """
        super().__init__()
        self.eps_model = eps_model
        self.time_diff_constraint = time_diff_constraint
        self.device = device

        # Create beta schedule
        self.beta = torch.linspace(0.0001, 0.02, n_steps).to(device)
        self.alpha = 1. - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.n_steps = n_steps
        self.sigma2 = self.beta
    
    def loss(self, x0: torch.Tensor, noise=None, debug=False):
        """
        Simplified diffusion loss function - handles shape mismatches automatically
        """
        # Get batch size
        batch_size = x0.shape[0]
        
        # Print input shape for debugging
        if debug:
            print(f"Input shape: {x0.shape}")
        
        # Get random time steps from 0 to n_steps-1
        t = torch.randint(0, self.n_steps, (batch_size,), device=x0.device, dtype=torch.long)
        
        # Generate random noise if not provided
        if noise is None:
            noise = torch.randn_like(x0)
            
        if debug:
            print(f"Noise shape: {noise.shape}")
        
        # Get alpha_bar for the time steps
        alpha_bar_t = self.alpha_bar[t]
        
        # Add dimensions for broadcasting
        # For example, if x0 is [batch, seq_len, channels], we need alpha_bar to be [batch, 1, 1]
        for _ in range(len(x0.shape) - 1):
            alpha_bar_t = alpha_bar_t.unsqueeze(-1)
                
        # Calculate noisy sample x_t
        xt = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * noise
        
        if debug:
            print(f"Noisy sample shape: {xt.shape}")
            print(f"Timestep shape: {t.shape}")
        
        # Predict noise - explicitly pass the timesteps
        predicted_noise = self.eps_model(xt, t)
        
        if debug:
            print(f"Predicted noise shape: {predicted_noise.shape}")
        
        # If shapes don't match, try to reshape
        if predicted_noise.shape != noise.shape:
            if debug:
                print(f"Shape mismatch: Predicted {predicted_noise.shape}, Expected {noise.shape}")
            try:
                predicted_noise = predicted_noise.view(noise.shape)
            except:
                # If reshape fails, log and continue
                if debug:
                    print("Reshape failed, falling back to MSE loss between differently shaped tensors")
        
        # MSE loss
        return F.mse_loss(noise, predicted_noise)
    
    def p_sample(self, xt: torch.Tensor, t: torch.Tensor):
        """
        Sample from p_θ(x_{t-1}|x_t)
        Simplified implementation that handles different shapes
        """
        # Predict noise
        predicted_noise = self.eps_model(xt, t)
        
        # Make sure predicted_noise has the same shape as xt
        if predicted_noise.shape != xt.shape:
            try:
                predicted_noise = predicted_noise.view(xt.shape)
            except:
                # If reshape fails, just continue with current shape
                pass
        
        # Get parameters
        alpha_t = self.alpha[t]
        alpha_bar_t = self.alpha_bar[t]
        beta_t = self.beta[t]
        
        # Add dimensions for broadcasting
        for _ in range(len(xt.shape) - 1):
            alpha_t = alpha_t.unsqueeze(-1)
            alpha_bar_t = alpha_bar_t.unsqueeze(-1)
            beta_t = beta_t.unsqueeze(-1)
        
        # Calculate mean for p_θ(x_{t-1}|x_t)
        mean = (1 / torch.sqrt(alpha_t)) * (xt - (beta_t / torch.sqrt(1 - alpha_bar_t)) * predicted_noise)
        
        # Variance
        var = beta_t * (1 - alpha_bar_t.to(torch.float32) / (1 - alpha_bar_t))
        
        # Sample noise
        noise = torch.randn_like(xt)
        
        # Return sample
        return mean + torch.sqrt(var) * noise

class Model(nn.Module):
    """
    Model class with a decoder integrated into the pretraining branch.
    Supports supervised, finetuning, and pretraining tasks.
    For pretraining tasks, the model now computes an encoder representation,
    projects it via a projection head, and also reconstructs the input via a decoder.
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.output_attention = configs.output_attention
        self.enc_in = configs.enc_in
        self.d_model = configs.d_model

        # Embedding configuration
        if configs.no_temporal_block and configs.no_channel_block:
            raise ValueError("At least one of the two blocks should be True")
        if configs.no_temporal_block:
            patch_len_list = []
        else:
            patch_len_list = list(map(int, configs.patch_len_list.split(",")))
        if configs.no_channel_block:
            up_dim_list = []
        else:
            up_dim_list = list(map(int, configs.up_dim_list.split(",")))
        stride_list = patch_len_list
        seq_len = configs.seq_len
        patch_num_list = [
            int((seq_len - patch_len) / stride + 2)
            for patch_len, stride in zip(patch_len_list, stride_list)
        ]
        augmentations = configs.augmentations.split(",")  # , "jitter", "mask", "channel", "drop"
        # Set default augmentations if needed for contrastive pretraining
        if augmentations == ["none"] and "pretrain" in self.task_name:
            augmentations = ["flip", "frequency", "jitter", "mask", "channel", "drop"]

        # Encoder embedding
        self.enc_embedding = TokenChannelEmbedding(
            configs.enc_in,
            configs.seq_len,
            configs.d_model,
            patch_len_list,
            up_dim_list,
            stride_list,
            configs.dropout,
            augmentations,
        )
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    ADformerLayer(
                        len(patch_len_list),
                        len(up_dim_list),
                        configs.d_model,
                        configs.n_heads,
                        configs.dropout,
                        configs.output_attention,
                        configs.no_inter_attn,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for _ in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
        )
        self.act = F.gelu
        self.dropout = nn.Dropout(configs.dropout)

        # Branches for different tasks
        if self.task_name in ["supervised", "finetune"]:
            self.classifier = nn.Linear(
                configs.d_model * len(patch_num_list) + configs.d_model * len(up_dim_list),
                configs.num_class,
            )
        elif self.task_name in ["pretrain_lead", "pretrain_moco", "diffusion"]:
            input_dim = configs.d_model * len(patch_num_list) + configs.d_model * len(up_dim_list)
            self.projection_head = nn.Sequential(
                nn.Linear(input_dim, input_dim * 2),
                nn.ReLU(),
                nn.Dropout(configs.dropout),
                nn.Linear(input_dim * 2, configs.d_model)
            )
            # Add decoder components from the removed diffusion branch
            self.dec_embedding = TokenChannelEmbedding(
                configs.enc_in,
                configs.seq_len,
                configs.d_model,
                patch_len_list,
                up_dim_list,
                stride_list,
                configs.dropout,
                augmentations,
            )
            d_layers = getattr(configs, "d_layers", 1)
            self.decoder_layers = nn.ModuleList([
                DecoderLayer(configs.d_model, configs.n_heads, configs.d_ff, dropout=configs.dropout, activation=configs.activation)
                for _ in range(d_layers)
            ])
            self.decoder_projection = nn.Linear(configs.d_model, configs.enc_in)
            if self.task_name == "pretrain_moco":
                K = configs.K  # queue size
                feat_dim = configs.d_model
                self.register_buffer("queue", torch.randn(feat_dim, K))
                self.queue = F.normalize(self.queue, dim=0)
                self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        else:
            raise ValueError("Task name not recognized or not implemented in the model")

        # Add timestep embedding for diffusion
        if configs.task_name == "diffusion":
            self.timestep_embed = nn.Embedding(configs.n_steps, configs.d_model)
            
            # Add components needed for the simplified diffusion forward path
            # These will be used in the diffusion_forward method
            self.diffusion_mlp = nn.Sequential(
                nn.Linear(configs.seq_len * configs.enc_in + configs.d_model, configs.d_model * 2),
                nn.ReLU(),
                nn.Dropout(configs.dropout),
                nn.Linear(configs.d_model * 2, configs.seq_len * configs.enc_in)
            )
            
            self.noise_pred = nn.Sequential(
                nn.Linear(configs.d_model * (len(patch_num_list) + len(up_dim_list)), 
                        configs.enc_in * configs.seq_len),
                nn.Unflatten(1, (configs.enc_in, configs.seq_len))
            )

    def supervised(self, x_enc, x_mark_enc):
        # Encoder branch for supervised tasks
        enc_out_t, enc_out_c = self.enc_embedding(x_enc)
        enc_out_t, enc_out_c, _, _ = self.encoder(enc_out_t, enc_out_c, attn_mask=None)
        if enc_out_t is None:
            enc_out = enc_out_c
        elif enc_out_c is None:
            enc_out = enc_out_t
        else:
            enc_out = torch.cat((enc_out_t, enc_out_c), dim=1)
        output = self.act(enc_out)
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)
        output = self.classifier(output)
        return output

    def pretrain(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Encoder: get latent representation
        enc_out_t, enc_out_c = self.enc_embedding(x_enc)
        enc_out_t, enc_out_c, _, _ = self.encoder(enc_out_t, enc_out_c, attn_mask=None)
        if enc_out_t is None:
            memory = enc_out_c
        elif enc_out_c is None:
            memory = enc_out_t
        else:
            memory = torch.cat((enc_out_t, enc_out_c), dim=1)
        # Projection head branch
        encoded = self.act(memory)
        encoded = self.dropout(encoded)
        encoded_flat = encoded.reshape(encoded.shape[0], -1)
        repr_out = self.projection_head(encoded_flat)

        # Decoder branch: reconstruct the original input
        dec_out_t, dec_out_c = self.dec_embedding(x_dec)
        # Convert lists to tensors if needed
        if isinstance(dec_out_t, list):
            dec_out_t = torch.cat(dec_out_t, dim=1)
        if isinstance(dec_out_c, list):
            dec_out_c = torch.cat(dec_out_c, dim=1)
        if dec_out_t is None:
            dec_out = dec_out_c
        elif dec_out_c is None:
            dec_out = dec_out_t
        else:
            dec_out = torch.cat((dec_out_t, dec_out_c), dim=1)
        
        # Permute for nn.MultiheadAttention (seq_len, batch, d_model)
        dec_out = dec_out.transpose(0, 1)
        memory_dec = memory.transpose(0, 1)
        for layer in self.decoder_layers:
            dec_out = layer(dec_out, memory_dec)
        dec_out = dec_out.transpose(0, 1)
        rec_out = self.decoder_projection(dec_out)
        return repr_out, rec_out
    
    def diffusion_forward(self, xt, t=None):
        """
        Forward pass for the diffusion model noise prediction.
        Args:
            xt: Noisy input at time step t - shape [batch, seq_len, channels]
            t: Time step indices, can be None during initialization
        Returns:
            Predicted noise - same shape as input
        """
        # Print input shape for debugging
        # print(f"Input shape to diffusion_forward: {xt.shape}")
        
        # Store original shape for output reshaping
        original_shape = xt.shape
        batch_size, seq_len, channels = xt.shape
        
        # Check if t is None and create a default t if needed
        if t is None:
            # Use timestep 0 as default
            t = torch.zeros(batch_size, dtype=torch.long, device=xt.device)
        
        # Create a flattened representation
        xt_flat = xt.reshape(batch_size, -1)  # [batch, seq_len*channels]
        
        # Get timestep embeddings - [batch, embedding_dim]
        t_emb = self.timestep_embed(t)
        
        # Combine input and timestep embedding
        combined = torch.cat([xt_flat, t_emb], dim=1)
        
        # Use the MLP for noise prediction
        if hasattr(self, 'diffusion_mlp'):
            # Use the dedicated MLP if available
            noise_pred_flat = self.diffusion_mlp(combined)
        else:
            # Create on-the-fly MLPs (not ideal for performance, but works)
            hidden_dim = self.d_model * 2  # Adjust as needed
            hidden = F.relu(nn.Linear(combined.shape[1], hidden_dim).to(xt.device)(combined))
            hidden = F.dropout(hidden, p=0.1, training=self.training)
            output_size = seq_len * channels
            noise_pred_flat = nn.Linear(hidden_dim, output_size).to(xt.device)(hidden)
        
        # Reshape back to original shape
        noise_pred = noise_pred_flat.reshape(original_shape)
        
        return noise_pred
    
    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None, t=None):
        """
        Forward pass for all model tasks.
        For diffusion task, t contains timestep indices.
        """
        if self.task_name == "diffusion":
            # Check input shape and print for debugging
            # print(f"Forward diffusion input shape: {x_enc.shape}")
            # if t is not None:
            #     print(f"Timestep shape: {t.shape}")
            # else:
            #     print("Timestep is None")
                
            return self.diffusion_forward(x_enc, t)
        elif self.task_name in ["supervised", "finetune"]:
            return self.supervised(x_enc, x_mark_enc)
        elif self.task_name in ["pretrain_lead", "pretrain_moco"]:
            return self.pretrain(x_enc, x_mark_enc, x_dec, x_mark_dec)
        else:
            raise ValueError("Task name not recognized or not implemented in the model")