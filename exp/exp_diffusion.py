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
        
        print("Training data shape:", train_data.X.shape)
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
        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
        
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, padding_mask) in enumerate(tqdm(train_loader)):
                iter_count += 1
                
                model_optim.zero_grad()
                
                batch_x = batch_x.float().to(self.device)
                
                # Get diffusion model (handles DataParallel correctly)
                diffusion = self.get_diffusion()
                
                # Training step with optional mixed precision
                if self.args.use_amp:
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
            self.model.load_state_dict(torch.load(model_path))
        
        # Initialize metrics dictionaries
        sample_val_metrics_dict = {"Accuracy": 0}
        subject_val_metrics_dict = {"Accuracy": 0}
        sample_test_metrics_dict = {"Accuracy": 0}
        subject_test_metrics_dict = {"Accuracy": 0}
        
        self.model.eval()
        
        # Get diffusion model (handles DataParallel correctly)
        diffusion = self.get_diffusion()
        
        # Sample new data from the diffusion model
        with torch.no_grad():
            n_samples = 16  # Number of samples to generate
            
            # Start with random noise
            x = torch.randn([n_samples, self.args.enc_in, self.args.seq_len], device=self.device)
            
            # Denoise step by step
            for t_ in tqdm(range(self.args.sample_steps), desc="Sampling"):
                t = self.args.n_steps - t_ - 1
                # Sample from p_θ(x_{t-1}|x_t)
                x = diffusion.p_sample(x, x.new_full((n_samples,), t, dtype=torch.long))
            
            # x now contains generated samples
            samples = x.cpu().numpy()
            
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
            
            np.save(output_path + "generated_samples.npy", samples)
            print(f"Generated samples saved to {output_path}generated_samples.npy")
        
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
                # Generate new samples from random noise
                n_samples = 16 if not hasattr(self.args, 'n_samples') else self.args.n_samples
                
                # Start with random noise
                x = torch.randn([n_samples, self.args.enc_in, self.args.seq_len], device=self.device)
                
                # Denoise step by step
                for t_ in tqdm(range(self.args.sample_steps), desc="Sampling"):
                    t = self.args.n_steps - t_ - 1
                    # Sample from p_θ(x_{t-1}|x_t)
                    x = diffusion.p_sample(x, x.new_full((n_samples,), t, dtype=torch.long))
                
                # Return generated samples
                return x.cpu().numpy()
                
            elif mode == "reconstruct":
                # Reconstruct inputs by adding noise and then denoising
                pred_samples = []
                
                for i, (batch_x, batch_y, padding_mask) in enumerate(test_loader):
                    batch_x = batch_x.float().to(self.device)
                    
                    # Add noise to the maximum step
                    t = torch.full((batch_x.shape[0],), self.args.n_steps-1, device=self.device, dtype=torch.long)
                    noisy_x = diffusion.q_sample(batch_x, t)
                    
                    # Denoise step by step
                    x = noisy_x
                    for t_ in tqdm(range(self.args.sample_steps), desc=f"Reconstructing batch {i+1}"):
                        t = self.args.n_steps - t_ - 1
                        # Sample from p_θ(x_{t-1}|x_t)
                        x = diffusion.p_sample(x, x.new_full((x.shape[0],), t, dtype=torch.long))
                    
                    pred_samples.append(x.cpu().numpy())
                
                # Concatenate all batches
                pred_samples = np.concatenate(pred_samples, axis=0)
                return pred_samples
            
            else:
                raise ValueError(f"Unknown mode: {mode}. Use 'sample' or 'reconstruct'.")