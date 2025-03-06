import os
import time
import warnings
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import random

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import LEAD  # Import LEAD model
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop


class Exp_Diffusion(Exp_Basic):
    def __init__(self, args):
        super(Exp_Diffusion, self).__init__(args)
        self.device = self._acquire_device()
        
    def _build_model(self):
        """Build the LEAD model with diffusion capabilities."""
        model = LEAD.Model(self.args).float()
        
        # Initialize the diffusion process
        model.diffusion = LEAD.DenoiseDiffusion(
            eps_model=model,  # We're using the model itself as the noise predictor
            n_steps=self.args.n_steps,
            device=self.device,
            time_diff_constraint=getattr(self.args, 'time_diff_constraint', True)
        )
        
        # If specified, load pretrained LEAD weights
        if hasattr(self.args, 'checkpoints_path') and self.args.checkpoints_path:
            print(f"Loading pretrained weights from {self.args.checkpoints_path}")
            try:
                checkpoint = torch.load(self.args.checkpoints_path, map_location=self.device)
                # Load only encoder/embedding weights, not the entire model
                model_dict = model.state_dict()
                
                # Handle different checkpoint formats
                pretrained_dict = {}
                if isinstance(checkpoint, dict) and 'model' in checkpoint:
                    pretrained_dict = {k: v for k, v in checkpoint['model'].items() 
                                      if 'enc_' in k or 'embedding' in k}
                else:
                    pretrained_dict = {k: v for k, v in checkpoint.items() 
                                      if 'enc_' in k or 'embedding' in k}
                
                model_dict.update(pretrained_dict)
                model.load_state_dict(model_dict, strict=False)
                print("Successfully loaded pretrained encoder weights")
            except Exception as e:
                print(f"Failed to load pretrained weights: {e}")
                print("Continuing with randomly initialized weights")
        
        # Create a reference to the diffusion model before wrapping in DataParallel
        self.diffusion_model = model.diffusion
                
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
            
        return model

    def _get_data(self, flag):
        """Get data loader for diffusion training/testing."""
        random.seed(self.args.seed)
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        """Select optimizer for diffusion training."""
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        """No explicit criterion needed for diffusion as it's handled in the model."""
        return None
        
    def get_diffusion(self):
        """Access the diffusion model correctly, regardless of DataParallel wrapping."""
        if isinstance(self.model, nn.DataParallel):
            # If model is wrapped with DataParallel, use the reference we created
            return self.diffusion_model
        else:
            # If not using DataParallel, access it directly
            return self.model.diffusion

    def train(self, setting):
        """Train the diffusion model."""
        train_data, train_loader = self._get_data(flag='TRAIN')
        vali_data, vali_loader = self._get_data(flag='VAL')
        test_data, test_loader = self._get_data(flag='TEST')
        
        print("Training data shape:", train_data.X.shape if hasattr(train_data, 'X') else "Unknown")
        if hasattr(train_data, 'y'):
            print("Training label shape:", train_data.y.shape)
        
        # Use the same path construction as in exp_pretrain.py
        path = (
            "./checkpoints/"
            + self.args.method
            + "/"
            + self.args.task_name
            + "/"
            + self.args.model
            + "/"
            + self.args.model_id
            + "/"
            + setting
            + "/"
        )
        
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        
        model_optim = self._select_optimizer()
        
        # Use automatic mixed precision for faster training if specified
        if hasattr(self.args, 'use_amp') and self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
        
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, padding_mask) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.args.train_epochs}")):
                iter_count += 1
                
                model_optim.zero_grad()
                
                batch_x = batch_x.float().to(self.device)
                
                # Get diffusion model (handles DataParallel correctly)
                diffusion = self.get_diffusion()
                
                # Print shape info for first batch of first epoch
                if epoch == 0 and i == 0:
                    print(f"Input batch shape: {batch_x.shape}")
                
                # Training step with optional mixed precision
                if hasattr(self.args, 'use_amp') and self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        loss = diffusion.loss(batch_x)
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss = diffusion.loss(batch_x)
                    loss.backward()
                    model_optim.step()
                
                train_loss.append(loss.item())
                
                if (i + 1) % 100 == 0:
                    print(f"\titers: {i + 1}/{train_steps}, epoch: {epoch + 1}, loss: {np.mean(train_loss):.7f}")
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print(f'\tspeed: {speed:.4f}s/iter, left time: {left_time:.4f}s')
                    iter_count = 0
                    time_now = time.time()
            
            print(f"Epoch: {epoch + 1}, Steps: {train_steps}, Train Loss: {np.mean(train_loss):.7f}")
            train_loss = np.average(train_loss)
            
            # Validation step
            vali_loss = self.validate(vali_loader)
            test_loss = self.validate(test_loader)
            
            print(f"Epoch: {epoch + 1}, Steps: {train_steps}, Vali Loss: {vali_loss:.7f}, Test Loss: {test_loss:.7f}")
            
            adjust_learning_rate(model_optim, epoch + 1, self.args)
            
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping triggered")
                break
            
            print(f"Epoch time: {time.time() - epoch_time:.2f}s")
            
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        
        return self.model

    def validate(self, data_loader):
        """Validate the diffusion model."""
        self.model.eval()
        total_loss = []
        
        # Get diffusion model (handles DataParallel correctly)
        diffusion = self.get_diffusion()
        
        with torch.no_grad():
            for i, (batch_x, batch_y, padding_mask) in enumerate(data_loader):
                batch_x = batch_x.float().to(self.device)
                loss = diffusion.loss(batch_x)
                total_loss.append(loss.item())
                
        return np.average(total_loss)

    def test(self, setting, test=0):
        """Test the diffusion model by sampling new data."""
        if test:
            print('loading model')
            # Use the same path construction for testing
            path = (
                "./checkpoints/"
                + self.args.method
                + "/"
                + self.args.task_name
                + "/"
                + self.args.model
                + "/"
                + self.args.model_id
                + "/"
                + setting
                + "/"
            )
            model_path = path + "checkpoint.pth"
            if os.path.exists(model_path):
                self.model.load_state_dict(torch.load(model_path))
                print(f"Successfully loaded model from {model_path}")
            else:
                print(f"Warning: Model file {model_path} not found. Using initialized model.")
        
        # Initialize metrics dictionaries
        sample_val_metrics_dict = {"Accuracy": 0}
        subject_val_metrics_dict = {"Accuracy": 0}
        sample_test_metrics_dict = {"Accuracy": 0}
        subject_test_metrics_dict = {"Accuracy": 0}
        
        self.model.eval()
        
        # Get diffusion model (handles DataParallel correctly)
        diffusion = self.get_diffusion()
        
        # Load test data to check input shapes
        test_data, test_loader = self._get_data(flag='TEST')
        
        # Try to determine the expected shape format from the dataset
        expected_shape_format = None
        if hasattr(test_data, 'X'):
            if isinstance(test_data.X, torch.Tensor) or isinstance(test_data.X, np.ndarray):
                data_shape = test_data.X.shape
                if len(data_shape) >= 3:
                    print(f"Dataset shape: {data_shape}")
                    # Determine if the format is (batch, seq_len, channels) or (batch, channels, seq_len)
                    # For EEG data, typically channels < seq_len
                    if data_shape[1] < data_shape[2]:
                        expected_shape_format = "batch_channel_seq"
                        print(f"Dataset appears to use (batch, channels, seq_len) format")
                    else:
                        expected_shape_format = "batch_seq_channel"
                        print(f"Dataset appears to use (batch, seq_len, channels) format")
        
        # Get number of samples to generate
        n_samples = getattr(self.args, 'num_samples', 16)
        samples_per_batch = getattr(self.args, 'samples_per_batch', 16)
        
        # Calculate number of batches needed
        num_batches = (n_samples + samples_per_batch - 1) // samples_per_batch
        
        print(f"Generating {n_samples} samples in {num_batches} batches")
        
        # Create a list to store all generated samples
        all_samples = []
        
        # Sample new data from the diffusion model in batches
        with torch.no_grad():
            for batch_idx in tqdm(range(num_batches), desc="Generating batches"):
                # Calculate batch size (may be smaller for the last batch)
                batch_size = min(samples_per_batch, n_samples - batch_idx * samples_per_batch)
                
                # Determine the shape to use for generation
                if expected_shape_format == "batch_channel_seq":
                    # Use (batch, channels, seq_len) format
                    sample_shape = (batch_size, self.args.enc_in, self.args.seq_len)
                    print(f"Using shape format: (batch, channels, seq_len) = {sample_shape}")
                else:
                    # Default to (batch, seq_len, channels) format
                    sample_shape = (batch_size, self.args.seq_len, self.args.enc_in)
                    print(f"Using shape format: (batch, seq_len, channels) = {sample_shape}")
                
                # Start with random noise
                x = torch.randn(sample_shape, device=self.device)
                
                # Store original shape to maintain consistency
                original_shape = x.shape
                
                # Denoise step by step
                for t_ in tqdm(range(self.args.sample_steps), desc=f"Sampling batch {batch_idx+1}/{num_batches}"):
                    t = self.args.n_steps - t_ - 1
                    
                    # Display shape before p_sample (first step only)
                    if t_ == 0:
                        print(f"Shape before p_sample: {x.shape}")
                    
                    # Sample from p_θ(x_{t-1}|x_t)
                    x = diffusion.p_sample(x, x.new_full((batch_size,), t, dtype=torch.long))
                    
                    # Ensure shape consistency after each step (important)
                    if x.shape != original_shape:
                        print(f"Shape changed during sampling. Adjusting from {x.shape} to {original_shape}")
                        # Try to reshape or transpose
                        try:
                            if x.dim() == 3 and original_shape[1] == x.shape[2] and original_shape[2] == x.shape[1]:
                                x = x.transpose(1, 2)
                            else:
                                x = x.reshape(batch_size, -1).reshape(original_shape)
                        except:
                            print(f"WARNING: Failed to maintain shape consistency during sampling")
                    
                    # Display shape after p_sample (first and last step)
                    if t_ == 0 or t_ == self.args.sample_steps-1:
                        print(f"Shape after p_sample (step {t_+1}): {x.shape}")
                
                # Add batch samples to our collection
                all_samples.append(x.cpu().numpy())
            
            # Combine all batches
            samples = np.concatenate(all_samples, axis=0)
            
            # Create result directory path
            output_path = (
                "./results/"
                + self.args.method
                + "/"
                + self.args.task_name
                + "/"
                + self.args.model
                + "/"
                + self.args.model_id
                + "/"
            )
            
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            
            # Save with description of number of samples
            filename = f"generated_samples_{n_samples}.npy"
            np.save(output_path + filename, samples)
            print(f"Generated {samples.shape[0]} samples with final shape {samples.shape}")
            print(f"Samples saved to {output_path}{filename}")
        
        # For compatibility with the evaluation framework
        params_count = test_params_flop(self.model)
        
        return sample_val_metrics_dict, subject_val_metrics_dict, sample_test_metrics_dict, subject_test_metrics_dict, params_count

    def predict(self, dataset, segment=False, mode="sample"):
        """
        Generate new EEG samples using the trained diffusion model.
        mode: "sample" for random sampling, "reconstruct" to denoise noisy inputs
        """
        self.model.eval()
        
        # Get diffusion model (handles DataParallel correctly)
        diffusion = self.get_diffusion()
        
        # If dataset provided, use it for conditional generation/reconstruction
        test_data, test_loader = self._get_data(flag='TEST')
        
        with torch.no_grad():
            if mode == "sample":
                # Get the size of the test dataset to match number of samples
                if hasattr(test_data, 'X'):
                    dataset_samples = test_data.X.shape[0]
                    print(f"Detected dataset size: {dataset_samples} samples")
                elif hasattr(test_data, 'x'):
                    dataset_samples = test_data.x.shape[0]
                    print(f"Detected dataset size: {dataset_samples} samples")
                else:
                    # If we can't determine dataset size, use default or arg value
                    dataset_samples = 546
                    print(f"Could not detect dataset size, using default: {dataset_samples} samples")
                
                # Get number of samples to generate (from args, or use dataset size)
                n_samples = getattr(self.args, 'num_samples', dataset_samples)
                # If arg was specifically provided, it overrides the dataset size
                if hasattr(self.args, 'num_samples') and self.args.num_samples > 0:
                    print(f"Overriding with specified number of samples: {n_samples}")
                
                samples_per_batch = getattr(self.args, 'samples_per_batch', 16)
                
                # Calculate number of batches needed
                num_batches = (n_samples + samples_per_batch - 1) // samples_per_batch
                
                print(f"Generating {n_samples} samples in {num_batches} batches (batch size: {samples_per_batch})")
                
                # Create a list to store all generated samples
                all_samples = []
                
                # Sample in batches
                for batch_idx in tqdm(range(num_batches), desc="Generating batches"):
                    # Calculate batch size (may be smaller for the last batch)
                    batch_size = min(samples_per_batch, n_samples - batch_idx * samples_per_batch)
                    
                    # Ensure shape is consistent with expected input format
                    # Using (batch_size, seq_len, channels) format to match data
                    sample_shape = (batch_size, self.args.seq_len, self.args.enc_in)
                    
                    # Start with random noise
                    x = torch.randn(sample_shape, device=self.device)
                    
                    # Store original shape to maintain consistency
                    original_shape = x.shape
                    
                    # Denoise step by step
                    for t_ in range(self.args.sample_steps):
                        t = self.args.n_steps - t_ - 1
                        # Sample from p_θ(x_{t-1}|x_t)
                        x = diffusion.p_sample(x, x.new_full((batch_size,), t, dtype=torch.long))
                        
                        # Ensure shape consistency
                        if x.shape != original_shape:
                            x = x.view(original_shape)
                    
                    # Add batch samples to our collection
                    all_samples.append(x.cpu().numpy())
                
                # Combine all batches
                samples = np.concatenate(all_samples, axis=0)
                print(f"Generated {samples.shape[0]} samples with shape {samples.shape}")
                
                return samples
                
            elif mode == "reconstruct":
                # Reconstruct inputs by adding noise and then denoising
                pred_samples = []
                
                for i, (batch_x, batch_y, padding_mask) in enumerate(test_loader):
                    batch_x = batch_x.float().to(self.device)
                    
                    # Store original shape
                    original_shape = batch_x.shape
                    
                    # Add noise to the maximum step
                    t = torch.full((batch_x.shape[0],), self.args.n_steps-1, device=self.device, dtype=torch.long)
                    noisy_x = diffusion.q_sample(batch_x, t)
                    
                    # Denoise step by step
                    x = noisy_x
                    for t_ in tqdm(range(self.args.sample_steps), desc=f"Reconstructing batch {i+1}"):
                        t = self.args.n_steps - t_ - 1
                        # Sample from p_θ(x_{t-1}|x_t)
                        x = diffusion.p_sample(x, x.new_full((x.shape[0],), t, dtype=torch.long))
                        
                        # Ensure shape consistency
                        if x.shape != original_shape:
                            x = x.view(original_shape)
                    
                    pred_samples.append(x.cpu().numpy())
                
                # Concatenate all batches
                pred_samples = np.concatenate(pred_samples, axis=0)
                print(f"Reconstructed {pred_samples.shape[0]} samples with shape {pred_samples.shape}")
                return pred_samples
            
            else:
                raise ValueError(f"Unknown mode: {mode}. Use 'sample' or 'reconstruct'.")