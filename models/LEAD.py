from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.ADformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import ADformerLayer
from layers.Embed import TokenChannelEmbedding
import math
import numpy as np

def frequency_constraint(signal, sampling_rate=128):
    """
    Apply frequency domain constraints to encourage realistic EEG rhythms.
    
    Args:
        signal: Input signal [batch, seq_len, channels] or [batch, channels, seq_len]
        sampling_rate: Sampling rate of the signal in Hz
        
    Returns:
        Signal with enhanced EEG frequency characteristics
    """
    # Check if we need to transpose for consistency
    signal_is_transposed = False
    if signal.dim() == 3 and signal.shape[2] > signal.shape[1]:
        # Assume [batch, channels, seq_len] format, transpose to [batch, seq_len, channels]
        signal = signal.transpose(1, 2)
        signal_is_transposed = True
    
    # Get dimensions
    batch_size, seq_len, n_channels = signal.shape
    
    # Process each channel separately
    processed_signal = []
    
    for ch in range(n_channels):
        # Extract channel data [batch, seq_len]
        channel_data = signal[:, :, ch]
        
        # Apply FFT
        fft = torch.fft.rfft(channel_data, dim=1)
        fft_mag = torch.abs(fft)
        fft_phase = torch.angle(fft)
        
        # Create frequency axis
        freqs = torch.fft.rfftfreq(seq_len, 1/sampling_rate, device=signal.device)
        
        # Create EEG rhythm masks
        delta_mask = ((freqs >= 0.5) & (freqs <= 4)).float() * 1.2  # Delta (0.5-4 Hz)
        theta_mask = ((freqs >= 4) & (freqs <= 8)).float() * 1.5    # Theta (4-8 Hz)
        alpha_mask = ((freqs >= 8) & (freqs <= 13)).float() * 2.0   # Alpha (8-13 Hz)
        beta_mask = ((freqs >= 13) & (freqs <= 30)).float() * 1.2   # Beta (13-30 Hz)
        gamma_mask = ((freqs >= 30) & (freqs <= 45)).float() * 0.8  # Gamma (30-45 Hz)
        
        # Set very high frequencies to lower values (common in EEG)
        high_freq_mask = (freqs > 45).float() * 0.3
        
        # Combine masks
        eeg_mask = delta_mask + theta_mask + alpha_mask + beta_mask + gamma_mask + high_freq_mask
        
        # Apply soft masks to enhance natural EEG characteristics
        # This doesn't completely remove frequencies but enhances the natural distribution
        weighted_fft_mag = fft_mag * (0.7 + 0.3 * eeg_mask.unsqueeze(0))
        
        # Reconstruct complex FFT using modified magnitude and original phase
        weighted_fft = weighted_fft_mag * torch.exp(1j * fft_phase)
        
        # Convert back to time domain
        processed_channel = torch.fft.irfft(weighted_fft, n=seq_len, dim=1)
        processed_signal.append(processed_channel.unsqueeze(-1))
    
    # Concatenate channels
    processed_signal = torch.cat(processed_signal, dim=-1)
    
    # Transpose back if needed
    if signal_is_transposed:
        processed_signal = processed_signal.transpose(1, 2)
    
    return processed_signal

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


class ArcMarginProduct(nn.Module):
    """
    Implement of large margin arc distance for subject classification.
    
    Args:
        in_features: Size of each input sample
        out_features: Size of each output sample (number of subjects)
        s: Norm of input feature
        m: Margin for angular distance
        easy_margin: Use easy margin version
    """
    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        
        # Initialize weight parameters
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        
        # Pre-compute constants for margin computation
        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
    
    def forward(self, input, label):
        """
        Forward pass with Arc Margin.
        
        Args:
            input: Feature vectors [batch, in_features]
            label: Subject labels [batch]
            
        Returns:
            Logits with margin applied [batch, out_features]
        """
        # Normalize input and weight
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        
        # Calculate sine
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        
        # Apply margin
        phi = cosine * self.cos_m - sine * self.sin_m
        
        # Apply margin conditions
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        
        # Apply one-hot encoding to labels
        one_hot = torch.zeros_like(cosine, device=input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        
        # Apply margin to target classes only
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        
        # Scale by s
        output = output * self.s
        
        return output


class ArcMarginHead(nn.Module):
    """
    ArcMargin classification head with feature projection layer.
    
    This simplified version doesn't depend on EEG_Net_8_Stack and can work with any input.
    
    Args:
        in_features: Size of input features
        hidden_features: Size of hidden layer (optional)
        out_features: Number of output classes (subjects)
        s: Scale factor for arc margin
        m: Margin for arc distance
    """
    def __init__(self, in_features, out_features, hidden_features=None, s=30.0, m=0.50, easy_margin=False):
        super(ArcMarginHead, self).__init__()
        
        # Optional feature projection layer
        self.use_projection = hidden_features is not None
        if self.use_projection:
            self.projection = nn.Sequential(
                nn.Linear(in_features, hidden_features),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_features)
            )
            arc_input_dim = hidden_features
        else:
            arc_input_dim = in_features
        
        # Arc margin classification layer
        self.arcpro = ArcMarginProduct(
            arc_input_dim, 
            out_features, 
            s=s, 
            m=m, 
            easy_margin=easy_margin
        )
    
    def forward(self, x, label):
        """
        Forward pass through projection and arcmargin.
        
        Args:
            x: Input features [batch, in_features]
            label: Subject labels [batch]
            
        Returns:
            Classification logits with margin [batch, out_features]
        """
        # Optional feature projection
        if self.use_projection:
            # Make sure x is 2D
            orig_shape = x.shape
            if x.dim() > 2:
                x = x.reshape(x.size(0), -1)
            
            # Project features
            x = self.projection(x)
        
        # Apply ArcMargin classification
        output = self.arcpro(x, label)
        
        return output


class SubjectNoisePredictor(nn.Module):
    """
    Subject-specific noise prediction network.
    
    This network takes the noisy EEG signal and timestep as input,
    and outputs a subject-specific noise prediction.
    """
    def __init__(self, configs):
        super(SubjectNoisePredictor, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.enc_in = configs.enc_in
        self.d_model = configs.d_model
        self.num_subjects = getattr(configs, 'num_subjects', 9)
        
        # Subject embedding
        self.subject_embedding = nn.Embedding(self.num_subjects, self.d_model)
        
        # Timestep embedding
        self.timestep_embed = nn.Embedding(configs.n_steps, self.d_model)
        
        # Encoder embedding
        patch_len_list = list(map(int, configs.patch_len_list.split(",")))
        up_dim_list = list(map(int, configs.up_dim_list.split(",")))
        stride_list = patch_len_list
        augmentations = configs.augmentations.split(",")
        
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
        
        # Encoder for subject noise prediction
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
        
        # Projection for noise prediction
        self.noise_projection = nn.Linear(
            configs.d_model * len(patch_len_list) + configs.d_model * len(up_dim_list),
            configs.enc_in * configs.seq_len
        )
        
        # MLP for combining embeddings
        self.combination_mlp = nn.Sequential(
            nn.Linear(self.d_model * 2, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, self.d_model)
        )
        
    def forward(self, x, t, s):
        """
        Forward pass for subject-specific noise prediction.
        
        Args:
            x: Noisy input at time step t [batch, seq_len, channels]
            t: Timestep indices [batch]
            s: Subject indices [batch]
            
        Returns:
            subject_mu: Subject-specific embeddings
            subject_theta: Subject-specific noise prediction
        """
        batch_size = x.shape[0]
        
        # Get subject embedding
        subject_emb = self.subject_embedding(s)  # [batch, d_model]
        
        # Get timestep embedding
        time_emb = self.timestep_embed(t)  # [batch, d_model]
        
        # Combine subject and time embeddings
        combined_emb = self.combination_mlp(
            torch.cat([subject_emb, time_emb], dim=1)
        )  # [batch, d_model]
        
        # Reshape input if needed for encoder
        if x.dim() == 3:
            # [batch, seq_len, channels] -> [batch, channels, seq_len]
            x = x.permute(0, 2, 1)
        
        # Get encoder embeddings
        enc_out_t, enc_out_c = self.enc_embedding(x)
        
        # Add subject-time embedding to encoder input
        if isinstance(enc_out_t, list):
            for i in range(len(enc_out_t)):
                enc_out_t[i] = enc_out_t[i] + combined_emb.unsqueeze(1)
        else:
            enc_out_t = enc_out_t + combined_emb.unsqueeze(1)
            
        if isinstance(enc_out_c, list):
            for i in range(len(enc_out_c)):
                enc_out_c[i] = enc_out_c[i] + combined_emb.unsqueeze(1)
        else:
            enc_out_c = enc_out_c + combined_emb.unsqueeze(1)
        
        # Process through encoder
        enc_out_t, enc_out_c, _, _ = self.encoder(enc_out_t, enc_out_c, attn_mask=None)
        
        # Combine encoder outputs
        if enc_out_t is None:
            memory = enc_out_c
        elif enc_out_c is None:
            memory = enc_out_t
        else:
            memory = torch.cat((enc_out_t, enc_out_c), dim=1)
        
        # Create subject-specific noise prediction
        memory_flat = memory.reshape(batch_size, -1)
        subject_theta_flat = self.noise_projection(memory_flat)
        
        # Reshape to match input dimensions
        subject_theta = subject_theta_flat.reshape(x.shape)
        
        # Return both embeddings and noise prediction
        return combined_emb, subject_theta


class DenoiseDiffusion:
    """
    Denoising Diffusion Probabilistic Model with subject-specific conditioning.
    
    This implements the DDPM algorithm as described in 
    "Denoising Diffusion Probabilistic Models" by Ho et al. (2020)
    with extensions for subject-specific EEG generation.
    """
    def __init__(self, eps_model: nn.Module, n_steps: int, device: torch.device, 
                 sub_theta: nn.Module = None, sub_arc_head: nn.Module = None,
                 time_diff_constraint=True, debug=False):
        """
        Initialize the diffusion process with optional subject-specific components.
        
        Args:
            eps_model: The model that predicts content noise
            n_steps: Number of diffusion steps
            device: The device to place constants on
            sub_theta: Optional subject-specific noise prediction network
            sub_arc_head: Optional ArcMargin classification head for subjects
            time_diff_constraint: Whether to use time difference constraint
            debug: Whether to print debug information
        """
        super().__init__()
        self.eps_model = eps_model
        self.sub_theta = sub_theta
        self.sub_arc_head = sub_arc_head
        self.time_diff_constraint = time_diff_constraint
        self.device = device
        self.debug = debug
        # Add the frequency constraint function as an attribute
        self.frequency_constraint = frequency_constraint

        # Create beta schedule (linearly increasing variance)
        self.beta = torch.linspace(0.0001, 0.02, n_steps).to(device)
        
        # Alpha values (1 - beta)
        self.alpha = 1. - self.beta
        
        # Cumulative product of alpha values
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        
        # Number of timesteps
        self.n_steps = n_steps
        
        # Variance schedule
        self.sigma2 = self.beta
        
        # Parameters for time difference constraint
        self.step_size = 75
        self.window_size = 224
        
        # Number of possible subjects (needed for random sampling)
        self.subject_noise_range = 9
        
        if debug:
            print(f"Initialized diffusion model with {n_steps} steps")
            print(f"Beta range: {self.beta[0].item():.6f} to {self.beta[-1].item():.6f}")
            if sub_theta is not None:
                print("Subject-specific noise prediction network provided")
            if sub_arc_head is not None:
                print("Subject classification head provided")

    def q_xt_x0(self, x0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get q(x_t|x_0) distribution: the distribution of adding noise to x_0 to get x_t.
        
        Args:
            x0: Original clean data [batch, seq_len, channels]
            t: Timesteps [batch]
            
        Returns:
            mean: Mean of the conditional distribution q(x_t|x_0)
            var: Variance of the conditional distribution q(x_t|x_0)
        """
        # Get alpha_bar_t for the given timesteps
        alpha_bar_t = self.alpha_bar[t]
        
        # Add dimensions for broadcasting
        # For 3D input [batch, seq_len, channels], we need [batch, 1, 1]
        for _ in range(len(x0.shape) - 1):
            alpha_bar_t = alpha_bar_t.unsqueeze(-1)
        
        # Mean: √(α̅_t) * x_0
        mean = torch.sqrt(alpha_bar_t) * x0
        
        # Variance: (1 - α̅_t)
        var = 1 - alpha_bar_t
        
        return mean, var

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, eps: Optional[torch.Tensor] = None):
        """
        Sample from q(x_t|x_0): add noise to x_0 according to the diffusion schedule.
        
        Args:
            x0: Original clean data [batch, seq_len, channels]
            t: Timesteps [batch]
            eps: Optional pre-generated noise (if None, will be sampled)
            
        Returns:
            x_t: Noisy samples at timestep t
        """
        # Generate random noise if not provided
        if eps is None:
            eps = torch.randn_like(x0)
        
        # Get mean and variance of q(x_t|x_0)
        mean, var = self.q_xt_x0(x0, t)
        
        # Sample from q(x_t|x_0) = N(mean, var)
        # x_t = √(α̅_t) * x_0 + √(1 - α̅_t) * ε
        return mean + torch.sqrt(var) * eps

    def p_sample(self, xt: torch.Tensor, t: torch.Tensor):
        """
        Sample from p_θ(x_{t-1}|x_t): perform one denoising step with enhanced processing.
        
        Args:
            xt: Noisy input at time step t [batch, seq_len, channels] or [batch, channels, seq_len]
            t: Timestep indices [batch]
            
        Returns:
            x_{t-1}: Sample with less noise (one step of denoising)
        """
        # Store original shape for consistent output
        original_shape = xt.shape
        batch_size = xt.shape[0]
        
        # For early steps (high noise), nothing special needed
        # For later steps (getting close to final image), apply enhancements
        is_late_step = t[0] < self.n_steps * 0.3  # Consider last 30% as "late steps"
        is_very_late_step = t[0] < self.n_steps * 0.1  # Last 10% as "very late steps"
        is_final_steps = t[0] < self.n_steps * 0.05  # Last 5% as "final steps"
        
        # Predict noise using the model
        predicted_noise = self.eps_model(xt, t)
        
        # Make sure predicted_noise has the same shape as xt
        if predicted_noise.shape != xt.shape:
            try:
                # Try direct reshaping
                predicted_noise = predicted_noise.view(xt.shape)
            except:
                try:
                    # Try transposing if dimensions are swapped
                    if predicted_noise.dim() == 3 and xt.dim() == 3:
                        if predicted_noise.shape[1] == xt.shape[2] and predicted_noise.shape[2] == xt.shape[1]:
                            predicted_noise = predicted_noise.transpose(1, 2)
                    # If still doesn't match, try to reshape anyway
                    if predicted_noise.shape != xt.shape:
                        predicted_noise = predicted_noise.reshape(xt.shape[0], -1).reshape(xt.shape)
                except:
                    # If reshape fails, log a warning
                    print(f"Warning: Shape mismatch in p_sample. Predicted: {predicted_noise.shape}, Expected: {xt.shape}")
        
        # Get parameters from the noise schedule
        alpha_t = self.alpha[t]
        alpha_bar_t = self.alpha_bar[t]
        beta_t = self.beta[t]
        
        # Add dimensions for broadcasting
        for _ in range(len(xt.shape) - 1):
            alpha_t = alpha_t.unsqueeze(-1)
            alpha_bar_t = alpha_bar_t.unsqueeze(-1)
            beta_t = beta_t.unsqueeze(-1)
        
        # Calculate the mean for p_θ(x_{t-1}|x_t)
        # μ_θ(x_t, t) = (1/√α_t) * (x_t - (β_t/√(1-α̅_t)) * ε_θ(x_t, t))
        eps_coef = beta_t / torch.sqrt(1 - alpha_bar_t)
        mean = (1 / torch.sqrt(alpha_t)) * (xt - eps_coef * predicted_noise)
        
        # Variance
        # In DDPM paper, the variance is either fixed to β_t or learned
        # Here we use the fixed variance schedule
        var = beta_t
        
        # Sample noise
        noise = torch.randn_like(xt)
        
        # Only add noise if t > 0, otherwise just return the mean
        # This ensures x_0 is deterministic and not noisy
        is_last_step = (t == 0).view((-1,) + (1,) * (len(xt.shape) - 1))
        noise = torch.where(is_last_step, 0.0, noise)
        
        # Apply variance annealing for smoother outputs in later steps
        if is_late_step:
            # Reduce noise variance in late steps
            noise_strength = torch.ones_like(noise)
            # Create a factor that decreases as we get closer to t=0
            timestep_factor = t[0].float() / self.n_steps
            # Scale noise by this factor (more reduction in later steps)
            noise_strength = noise_strength * (0.5 + 0.5 * timestep_factor)
            noise = noise * noise_strength
        
        # Calculate result with possibly reduced noise
        result = mean + torch.sqrt(var) * noise
        
        # For very late steps, apply additional processing to enhance naturalness
        if is_very_late_step:
            # Apply temporal smoothing
            if result.dim() == 3:
                # Identify which dimension is time (usually the longer one)
                time_dim = 1 if result.shape[1] >= result.shape[2] else 2
                
                if time_dim == 1:
                    # Apply 1D temporal smoothing with a small kernel
                    kernel_size = min(5, result.shape[1] // 10)  # Kernel size based on signal length
                    if kernel_size % 2 == 0:  # Ensure odd kernel size
                        kernel_size += 1
                        
                    if kernel_size > 2:  # Only smooth if kernel size is reasonable
                        padding = kernel_size // 2
                        # Create a 1D smoothing kernel (gaussian-like)
                        smoothing_kernel = torch.tensor(
                            [0.1, 0.2, 0.4, 0.2, 0.1] if kernel_size == 5 else [1/kernel_size] * kernel_size,
                            device=result.device
                        ).view(1, 1, kernel_size)
                        
                        # Process each channel separately
                        channels = result.shape[2]
                        smoothed = []
                        for c in range(channels):
                            channel = result[:, :, c:c+1]  # [batch, time, 1]
                            # Apply convolution for smoothing
                            smoothed_channel = F.conv1d(
                                F.pad(channel.transpose(1, 2), (padding, padding), mode='replicate'),
                                smoothing_kernel,
                                groups=1
                            ).transpose(1, 2)
                            smoothed.append(smoothed_channel)
                        
                        # Combine smoothed channels
                        smoothed_result = torch.cat(smoothed, dim=2)
                        
                        # Blend original and smoothed based on timestep
                        # More smoothing as we approach t=0
                        blend_factor = 1.0 - (t[0].float() / (self.n_steps * 0.1))
                        result = (1.0 - blend_factor) * result + blend_factor * smoothed_result
                
                elif time_dim == 2:
                    # Similar smoothing but for [batch, channels, time] format
                    kernel_size = min(5, result.shape[2] // 10)
                    if kernel_size % 2 == 0:
                        kernel_size += 1
                        
                    if kernel_size > 2:
                        padding = kernel_size // 2
                        smoothing_kernel = torch.tensor(
                            [0.1, 0.2, 0.4, 0.2, 0.1] if kernel_size == 5 else [1/kernel_size] * kernel_size,
                            device=result.device
                        ).view(1, 1, kernel_size)
                        
                        # Process each channel
                        channels = result.shape[1]
                        smoothed = []
                        for c in range(channels):
                            channel = result[:, c:c+1, :]  # [batch, 1, time]
                            # Apply convolution
                            smoothed_channel = F.conv1d(
                                F.pad(channel, (padding, padding), mode='replicate'),
                                smoothing_kernel,
                                groups=1
                            )
                            smoothed.append(smoothed_channel)
                        
                        # Combine channels
                        smoothed_result = torch.cat(smoothed, dim=1)
                        
                        # Blend based on timestep
                        blend_factor = 1.0 - (t[0].float() / (self.n_steps * 0.1))
                        result = (1.0 - blend_factor) * result + blend_factor * smoothed_result
        
        # For final steps, apply frequency domain constraints for realistic EEG
        if is_final_steps and hasattr(self, 'frequency_constraint'):
            try:
                # Apply frequency domain enhancements
                freq_enhanced = self.frequency_constraint(result)
                
                # Blend original and frequency-enhanced based on timestep
                blend_factor = 1.0 - (t[0].float() / (self.n_steps * 0.05))
                result = (1.0 - blend_factor) * result + blend_factor * freq_enhanced
            except Exception as e:
                # Fallback if frequency processing fails
                print(f"Warning: Frequency processing failed: {e}")
        
        # Ensure output shape consistency
        if result.shape != original_shape:
            try:
                result = result.view(original_shape)
            except:
                try:
                    if result.dim() == 3 and original_shape[1] == result.shape[2] and original_shape[2] == result.shape[1]:
                        result = result.transpose(1, 2)
                    else:
                        result = result.reshape(batch_size, -1).reshape(original_shape)
                except:
                    print(f"Error: Failed to maintain shape consistency in p_sample")
        
        return result
        
    def p_sample_noise(self, xt: torch.Tensor, t: torch.Tensor, s: torch.Tensor):
        """
        Sample from p_θ(x_{t-1}|x_t) with subject-specific conditioning.
        
        Args:
            xt: Noisy input at time step t [batch, seq_len, channels]
            t: Timestep indices [batch]
            s: Subject indices for conditioning [batch]
            
        Returns:
            x_{t-1}: Sample with less noise, conditioned on subject
        """
        if not hasattr(self, 'sub_theta') or self.sub_theta is None:
            raise ValueError("Subject-specific network not available for conditional sampling")
        
        # Get subject-specific noise prediction
        subject_mu, subject_theta = self.sub_theta(xt, t, s)
        
        # Get parameters from the noise schedule
        alpha_t = self.alpha[t]
        alpha_bar_t = self.alpha_bar[t]
        beta_t = self.beta[t]
        
        # Add dimensions for broadcasting
        for _ in range(len(xt.shape) - 1):
            alpha_t = alpha_t.unsqueeze(-1)
            alpha_bar_t = alpha_bar_t.unsqueeze(-1)
            beta_t = beta_t.unsqueeze(-1)
        
        # Calculate the mean
        eps_coef = beta_t / torch.sqrt(1 - alpha_bar_t)
        mean = (1 / torch.sqrt(alpha_t)) * (xt - eps_coef * subject_theta)
        
        # Variance
        var = beta_t
        
        # Sample noise
        eps = torch.randn_like(xt)
        
        # Only add noise if t > 0
        is_last_step = (t == 0).view((-1,) + (1,) * (len(xt.shape) - 1))
        eps = torch.where(is_last_step, 0.0, eps)
        
        # Return sample
        return mean + torch.sqrt(var) * eps

    def loss(self, x0: torch.Tensor, noise: Optional[torch.Tensor] = None, debug=False):
        """
        Calculate the simplified diffusion loss:
        L_simple = E_t,x_0,ε[||ε - ε_θ(x_t, t)||^2]
        
        Args:
            x0: Original clean data [batch, seq_len, channels]
            noise: Optional pre-generated noise
            debug: Whether to print debug information
            
        Returns:
            loss: The MSE loss between predicted and actual noise
        """
        local_debug = debug or self.debug
        batch_size = x0.shape[0]
        
        if local_debug:
            print(f"Input shape: {x0.shape}")
        
        # Sample random timesteps
        t = torch.randint(0, self.n_steps, (batch_size,), device=x0.device, dtype=torch.long)
        
        # Generate random noise if not provided
        if noise is None:
            noise = torch.randn_like(x0)
            
        if local_debug:
            print(f"Noise shape: {noise.shape}")
        
        # Get noisy samples x_t at timestep t
        xt = self.q_sample(x0, t, eps=noise)
        
        if local_debug:
            print(f"Noisy sample shape: {xt.shape}")
            print(f"Timestep shape: {t.shape}")
        
        # Predict noise
        predicted_noise = self.eps_model(xt, t)
        
        if local_debug:
            print(f"Predicted noise shape: {predicted_noise.shape}")
        
        # Handle shape mismatch if necessary
        if predicted_noise.shape != noise.shape:
            if local_debug:
                print(f"Shape mismatch: Predicted {predicted_noise.shape}, Expected {noise.shape}")
            try:
                predicted_noise = predicted_noise.view(noise.shape)
            except:
                if local_debug:
                    print("Reshape failed, falling back to MSE loss between differently shaped tensors")
        
        # Calculate time difference constraint penalty if enabled
        constraint_penalty = 0
        if self.time_diff_constraint and len(x0.shape) >= 3 and x0.shape[1] > 1:
            # Example constraint: adjacent time steps should have similar noise predictions
            # This encourages temporal consistency in the generated signals
            time_dim = 1  # Usually time is dimension 1 in [batch, time, channels]
            for i in range(predicted_noise.shape[time_dim] - 1):
                constraint_penalty += F.mse_loss(
                    predicted_noise[:, i+1], 
                    predicted_noise[:, i], 
                    reduction='mean'
                )
            constraint_penalty /= (predicted_noise.shape[time_dim] - 1)
            
            if local_debug:
                print(f"Time difference constraint penalty: {constraint_penalty.item():.6f}")
        
        # MSE loss between predicted and actual noise
        noise_loss = F.mse_loss(noise, predicted_noise)
        
        # Add constraint with a weight if enabled
        total_loss = noise_loss
        if self.time_diff_constraint:
            constraint_weight = 0.1  # Adjust as needed
            total_loss = noise_loss + constraint_weight * constraint_penalty
        
        return total_loss

    def loss_with_diff_constraint(self, x0: torch.Tensor, label: torch.Tensor, 
                            noise: Optional[torch.Tensor] = None, debug=False, 
                            noise_content_kl_co=1.0, arc_subject_co=0.1, orgth_co=2.0):
        """
        Enhanced loss function with multiple constraints for better EEG generation.
        
        Args:
            x0: Original clean data [batch, seq_len, channels]
            label: Subject labels for classification [batch]
            noise: Optional pre-generated noise
            debug: Whether to print debug information
            noise_content_kl_co: Weight for KL divergence loss
            arc_subject_co: Weight for subject classification loss
            orgth_co: Weight for orthogonality loss
            
        Returns:
            total_loss: The combined loss value
            constraint_penalty: Time difference constraint penalty value
            noise_content_kl: KL divergence between content and subject-specific noise
            subject_arc_loss: Subject classification loss
            loss_orth: Orthogonality loss value
        """
        local_debug = debug or self.debug
        batch_size = x0.shape[0]
        
        # Get random timesteps
        t = torch.randint(0, self.n_steps, (batch_size,), device=x0.device, dtype=torch.long)
        
        # Use subject labels
        s = label
        
        if local_debug:
            print(f"Input shape: {x0.shape}")
            print(f"Timestep shape: {t.shape}")
            print(f"Label shape: {s.shape}")
        
        # Generate random noise if not provided
        if noise is None:
            noise = torch.randn_like(x0)
        
        # Get noisy samples x_t at timestep t
        xt = self.q_sample(x0, t, eps=noise)
        
        if local_debug:
            print(f"Noisy sample shape: {xt.shape}")
        
        # Get content-specific noise prediction
        eps_theta = self.eps_model(xt, t)
        
        # Get subject-specific noise prediction using the subject embedding network
        if hasattr(self, 'sub_theta') and self.sub_theta is not None:
            subject_mu, subject_theta = self.sub_theta(xt, t, s)
        else:
            # If no subject network is available, create zero tensor with same shape
            subject_theta = torch.zeros_like(eps_theta)
            subject_mu = torch.zeros_like(eps_theta)
        
        if local_debug:
            print(f"Content noise shape: {eps_theta.shape}")
            print(f"Subject noise shape: {subject_theta.shape}")
        
        # Calculate time difference constraint penalty - ENHANCED VERSION
        constraint_penalty = 0
        if self.time_diff_constraint and xt.dim() >= 3 and xt.shape[1] > 1:
            # For EEG data, typically time is dimension 1
            time_dim = 1
            step_size = getattr(self, 'step_size', 1)
            
            # Calculate multiple levels of temporal consistency
            # Short-term consistency (adjacent time steps)
            short_term_penalty = 0
            for i in range(eps_theta.shape[time_dim] - 1):
                short_term_penalty += F.mse_loss(
                    eps_theta[:, i+1], 
                    eps_theta[:, i], 
                    reduction='mean'
                )
            short_term_penalty /= (eps_theta.shape[time_dim] - 1)
            
            # Medium-term consistency (steps apart)
            medium_term_penalty = 0
            for i in range(eps_theta.shape[time_dim] - step_size):
                medium_term_penalty += F.mse_loss(
                    eps_theta[:, i+step_size], 
                    eps_theta[:, i], 
                    reduction='mean'
                )
            medium_term_penalty /= (eps_theta.shape[time_dim] - step_size)
            
            # Long-term consistency (overall pattern)
            if eps_theta.shape[time_dim] > 10:
                segment_size = eps_theta.shape[time_dim] // 5  # Divide into 5 segments
                long_term_penalty = 0
                for i in range(eps_theta.shape[time_dim] - segment_size):
                    if i % segment_size == 0:  # Only check between segments
                        long_term_penalty += F.mse_loss(
                            eps_theta[:, i+segment_size], 
                            eps_theta[:, i], 
                            reduction='mean'
                        )
                long_term_penalty /= (eps_theta.shape[time_dim] // segment_size)
            else:
                long_term_penalty = 0
            
            # Combine penalties with weights favoring short and medium-term consistency
            constraint_penalty = 0.5 * short_term_penalty + 0.3 * medium_term_penalty + 0.2 * long_term_penalty
        
        # Calculate orthogonality loss between content and subject components
        # This encourages content and subject noise to be independent
        if eps_theta.dim() >= 3:
            # Reshape for matrix multiplication
            eps_flat = eps_theta.reshape(eps_theta.shape[0]*eps_theta.shape[1], -1)
            subj_flat = subject_theta.reshape(subject_theta.shape[0]*subject_theta.shape[1], -1)
            
            # Matrix product
            organal_squad = torch.bmm(
                eps_flat.unsqueeze(1), 
                subj_flat.unsqueeze(2)
            ).squeeze()
            
            # Create mask to penalize non-diagonal elements
            diag_size = organal_squad.shape[0]
            ones = torch.ones(diag_size, device=organal_squad.device)
            diag = torch.eye(diag_size, device=organal_squad.device)
            
            # Compute orthogonality loss
            loss_orth = ((organal_squad * (ones - diag)) ** 2).mean()
        else:
            # Fallback if dimensions don't match
            loss_orth = torch.tensor(0.0, device=x0.device)
        
        # Calculate KL divergence between content and subject-specific noise distributions
        noise_content_kl = F.kl_div(
            F.log_softmax(eps_theta.reshape(-1, eps_theta.shape[-1]), dim=-1),
            F.softmax(subject_theta.reshape(-1, subject_theta.shape[-1]), dim=-1),
            reduction='mean'
        )
        
        # Calculate subject classification loss using ArcMargin if available
        if hasattr(self, 'sub_arc_head') and self.sub_arc_head is not None:
            # Reshape subject_theta for classification
            if subject_theta.dim() >= 3:
                # Adjust dimensions for ArcMargin input
                arc_input = subject_theta.permute(0, 2, 1).mean(dim=2)
            else:
                arc_input = subject_theta
                
            # Get logits from ArcMargin head
            subject_arc_logit = self.sub_arc_head(arc_input, s)
            
            # Calculate cross entropy loss
            subject_arc_loss = F.cross_entropy(subject_arc_logit, s.long())
        else:
            # If ArcMargin is not available, use zero loss
            subject_arc_loss = torch.tensor(0.0, device=x0.device)
        
        # Combine all loss components
        noise_prediction_loss = F.mse_loss(noise, eps_theta + subject_theta)
        
        # Apply weights to each component
        total_loss = noise_prediction_loss
        
        # Enhanced temporal consistency weight (increased from 0.1 to 0.5)
        if self.time_diff_constraint:
            constraint_weight = 0.5  # Increased weight for stronger temporal constraint
            total_loss = total_loss + constraint_weight * constraint_penalty
        
        if orgth_co > 0:
            total_loss = total_loss + orgth_co * loss_orth
        
        if arc_subject_co > 0 and hasattr(self, 'sub_arc_head'):
            total_loss = total_loss + arc_subject_co * subject_arc_loss
        
        if noise_content_kl_co > 0:
            total_loss = total_loss - noise_content_kl_co * noise_content_kl
        
        if local_debug:
            print(f"Noise prediction loss: {noise_prediction_loss.item():.6f}")
            print(f"Constraint penalty: {constraint_penalty.item() if isinstance(constraint_penalty, torch.Tensor) else constraint_penalty:.6f}")
            print(f"Orthogonality loss: {loss_orth.item():.6f}")
            print(f"Subject arc loss: {subject_arc_loss.item():.6f}")
            print(f"KL divergence: {noise_content_kl.item():.6f}")
            print(f"Total loss: {total_loss.item():.6f}")
        
        return total_loss, constraint_penalty, noise_content_kl, subject_arc_loss, loss_orth

    def sample(self, shape, sample_steps=None):
        """
        Generate new samples from the diffusion model.
        
        Args:
            shape: Shape of samples to generate [batch, seq_len, channels]
            sample_steps: Number of sampling steps (if None, uses n_steps)
            
        Returns:
            Samples generated by the diffusion process
        """
        if sample_steps is None:
            sample_steps = self.n_steps
            
        # Start from pure noise
        batch_size = shape[0]
        device = self.device
        
        # Sample x_T from standard normal distribution
        x = torch.randn(shape, device=device)
        
        # Store original shape to ensure consistent output format
        original_shape = x.shape
        
        # Progressively denoise x_t for t = T, T-1, ..., 1
        for t_step in range(sample_steps):
            # Current timestep (going backwards)
            t = self.n_steps - t_step - 1
            
            # Create a batch of timesteps
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            
            # Sample from p(x_{t-1} | x_t)
            with torch.no_grad():
                x = self.p_sample(x, t_batch)
                
                # Ensure shape consistency through each step
                if x.shape != original_shape:
                    x = x.view(original_shape)
                
        return x
    
    def sample_with_subject(self, shape, subject_ids, sample_steps=None):
        """
        Generate new samples conditioned on subject IDs.
        
        Args:
            shape: Shape of samples to generate [batch, seq_len, channels]
            subject_ids: Subject IDs to condition on [batch]
            sample_steps: Number of sampling steps (if None, uses n_steps)
            
        Returns:
            Samples generated by the subject-conditioned diffusion process
        """
        if not hasattr(self, 'sub_theta') or self.sub_theta is None:
            raise ValueError("Subject-specific network not available for conditional sampling")
            
        if sample_steps is None:
            sample_steps = self.n_steps
            
        # Start from pure noise
        batch_size = shape[0]
        device = self.device
        
        # Convert subject_ids to tensor if needed
        if not isinstance(subject_ids, torch.Tensor):
            subject_ids = torch.tensor(subject_ids, device=device)
        
        # Sample x_T from standard normal
        x = torch.randn(shape, device=device)
        
        # Store original shape to ensure consistent output format
        original_shape = x.shape
        
        # Progressively denoise with subject conditioning
        for t_step in range(sample_steps):
            # Current timestep (going backwards)
            t = self.n_steps - t_step - 1
            
            # Create a batch of timesteps
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            
            # Sample from p(x_{t-1} | x_t) with subject conditioning
            with torch.no_grad():
                x = self.p_sample_noise(x, t_batch, subject_ids)
                
                # Ensure shape consistency through each step
                if x.shape != original_shape:
                    x = x.view(original_shape)
                
        return x

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
            xt: Noisy input at time step t - shape [batch, seq_len, channels] or [batch, channels, seq_len]
            t: Time step indices, can be None during initialization
        Returns:
            Predicted noise - same shape as input
        """
        # Store original shape for output reshaping
        original_shape = xt.shape
        batch_size = xt.shape[0]
        
        # Fix input shape inconsistency
        # The model expects [batch, seq_len, channels] but might get [batch, channels, seq_len]
        # Check dimensions and transpose if needed
        if xt.dim() == 3:
            # If channels (enc_in) is much smaller than seq_len, we can detect the right format
            # In your case, channels=19, seq_len=128
            if xt.shape[1] == self.enc_in and xt.shape[2] == self.seq_len:
                # Input is [batch, channels, seq_len], transpose to [batch, seq_len, channels]
                print(f"Transposing input from shape {xt.shape} to match expected format")
                xt = xt.transpose(1, 2)
            # elif xt.shape[1] == self.seq_len and xt.shape[2] == self.enc_in:
            #     # Already in correct [batch, seq_len, channels] format
            #     print(f"Input shape {xt.shape} already in expected format")
            # else:
            #     # If dimensions don't clearly match expected values, log a warning
            #     print(f"WARNING: Unexpected input shape {xt.shape}. Expected [{batch_size}, {self.seq_len}, {self.enc_in}] or [{batch_size}, {self.enc_in}, {self.seq_len}]")
        
        # Check if t is None and create a default t if needed
        if t is None:
            # Use timestep 0 as default
            t = torch.zeros(batch_size, dtype=torch.long, device=xt.device)
        
        # Get timestep embeddings
        t_emb = self.timestep_embed(t)  # [batch, d_model]
        
        # Process through encoder
        enc_out_t, enc_out_c = self.enc_embedding(xt)
        
        # Inject timestep embedding into encoder outputs
        if isinstance(enc_out_t, list):
            for i in range(len(enc_out_t)):
                enc_out_t[i] = enc_out_t[i] + t_emb.unsqueeze(1)
        else:
            enc_out_t = enc_out_t + t_emb.unsqueeze(1)
            
        if isinstance(enc_out_c, list):
            for i in range(len(enc_out_c)):
                enc_out_c[i] = enc_out_c[i] + t_emb.unsqueeze(1)
        else:
            enc_out_c = enc_out_c + t_emb.unsqueeze(1)
        
        # Process through encoder
        enc_out_t, enc_out_c, _, _ = self.encoder(enc_out_t, enc_out_c, attn_mask=None)
        
        # Combine encoder outputs
        if enc_out_t is None:
            enc_out = enc_out_c
        elif enc_out_c is None:
            enc_out = enc_out_t
        else:
            enc_out = torch.cat((enc_out_t, enc_out_c), dim=1)
        
        # Apply activation and dropout
        output = self.act(enc_out)
        output = self.dropout(output)
        
        # Flatten and predict noise
        output_flat = output.reshape(batch_size, -1)
        noise_pred_flat = self.noise_pred(output_flat)
        
        # Reshape to original input shape
        # The noise_pred outputs [batch, channels, seq_len] or [batch, seq_len, channels]
        noise_pred = noise_pred_flat
        
        # Ensure the output has the same shape as the input
        if noise_pred.shape != original_shape:
            # print(f"Reshaping output from {noise_pred.shape} to match input shape {original_shape}")
            # Try direct reshaping
            try:
                noise_pred = noise_pred.view(original_shape)
            except:
                # If direct reshape fails, try transposing dimensions
                try:
                    if noise_pred.dim() == 3 and original_shape[1] == noise_pred.shape[2] and original_shape[2] == noise_pred.shape[1]:
                        noise_pred = noise_pred.transpose(1, 2)
                    else:
                        # Last resort: just reshape to match original shape's sizes
                        noise_pred = noise_pred.reshape(batch_size, -1).reshape(original_shape)
                except:
                    print(f"ERROR: Failed to reshape output to match input shape. Output shape: {noise_pred.shape}, Input shape: {original_shape}")
        
        return noise_pred
    
    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None, t=None):
        """
        Forward pass for all model tasks.
        For diffusion task, t contains timestep indices.
        """
        if self.task_name == "diffusion":
            return self.diffusion_forward(x_enc, t)
        elif self.task_name in ["supervised", "finetune"]:
            return self.supervised(x_enc, x_mark_enc)
        elif self.task_name in ["pretrain_lead", "pretrain_moco"]:
            return self.pretrain(x_enc, x_mark_enc, x_dec, x_mark_dec)
        else:
            raise ValueError("Task name not recognized or not implemented in the model")