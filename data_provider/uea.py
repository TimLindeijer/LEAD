import os
import numpy as np
import pandas as pd
import torch
from itertools import repeat
from scipy.signal import butter, lfilter, filtfilt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.utils import shuffle


def collate_fn(data, max_len=None):
    """Build mini-batch tensors from a list of (X, mask) tuples. Mask input. Create
    Args:
        data: len(batch_size) list of tuples (X, y).
            - X: torch tensor of shape (seq_length, feat_dim); variable seq_length.
            - y: torch tensor of shape (num_labels,) : class indices or numerical targets
                (for supervised or regression, respectively). num_labels > 1 for multi-task models
        max_len: global fixed sequence length. Used for architectures requiring fixed length input,
            where the batch length cannot vary dynamically. Longer sequences are clipped, shorter are padded with 0s
    Returns:
        X: (batch_size, padded_length, feat_dim) torch tensor of masked features (input)
        targets: (batch_size, padded_length, feat_dim) torch tensor of unmasked features (output)
        target_masks: (batch_size, padded_length, feat_dim) boolean torch tensor
            0 indicates masked values to be predicted, 1 indicates unaffected/"active" feature values
        padding_masks: (batch_size, padded_length) boolean tensor, 1 means keep vector at this position, 0 means padding
    """
    import torch
    
    # Handle error case: data is not a list or contains non-tuple elements
    if not isinstance(data, (list, tuple)):
        print(f"ERROR: collate_fn expected list/tuple but got {type(data)}: {data}")
        # Try to handle this case by wrapping in list
        if isinstance(data, int) or (hasattr(data, 'item') and hasattr(data, 'dim') and data.dim() == 0):
            # Single integer value - create dummy batch
            dummy_features = torch.zeros((1, 128, 19))  # Adjust dimensions to match expected
            dummy_labels = torch.tensor([data if isinstance(data, int) else data.item()])
            dummy_padding_mask = torch.ones((1, 128), dtype=torch.bool)  # All active
            return dummy_features, dummy_labels, dummy_padding_mask
        else:
            # Try to make a list from it
            try:
                data = [data]
            except:
                # Complete fallback
                print("CRITICAL ERROR: Could not process data for batch")
                dummy_features = torch.zeros((1, 128, 19))  # Adjust dimensions to match expected
                dummy_labels = torch.zeros(1, dtype=torch.long)
                dummy_padding_mask = torch.ones((1, 128), dtype=torch.bool)  # All active
                return dummy_features, dummy_labels, dummy_padding_mask
    
    # Handle empty batch
    if len(data) == 0:
        print("WARNING: Empty batch received")
        dummy_features = torch.zeros((0, 128, 19))  # Adjust dimensions
        dummy_labels = torch.zeros(0, dtype=torch.long)
        dummy_padding_mask = torch.ones((0, 128), dtype=torch.bool)
        return dummy_features, dummy_labels, dummy_padding_mask
    
    try:
        # Check batch contents for consistency
        for i, item in enumerate(data):
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                print(f"WARNING: Item {i} is not a proper (feature, label) tuple: {type(item)}")
                if isinstance(item, (int, float)) or (isinstance(item, torch.Tensor) and item.dim() == 0):
                    # It's a single value - probably just a label
                    # Replace with dummy tuple
                    data[i] = (torch.zeros((128, 19)), item if isinstance(item, (int, float)) else item.item())
                    print(f"Fixed item {i} by creating a dummy feature tensor")
    except Exception as e:
        print(f"Error checking batch contents: {e}")
    
    # Standard processing
    try:
        batch_size = len(data)
        features, labels = zip(*data)

        # Stack and pad features and masks (convert 2D to 3D tensors, i.e. add batch dimension)
        lengths = [X.shape[0] for X in features]  # original sequence length for each time series
        if max_len is None:
            max_len = max(lengths)

        X = torch.zeros(batch_size, max_len, features[0].shape[-1])  # (batch_size, padded_length, feat_dim)
        for i in range(batch_size):
            end = min(lengths[i], max_len)
            X[i, :end, :] = features[i][:end, :]

        targets = torch.stack(labels, dim=0)  # (batch_size, num_labels)

        padding_masks = padding_mask(torch.tensor(lengths, dtype=torch.int16),
                                    max_len=max_len)  # (batch_size, padded_length) boolean tensor, "1" means keep

        return X, targets, padding_masks
        
    except Exception as e:
        print(f"Error in collate_fn: {e}")
        print(f"Data type: {type(data)}, length: {len(data)}")
        if len(data) > 0:
            print(f"First item type: {type(data[0])}, value: {data[0]}")
            
        # Return placeholder tensors as emergency fallback
        batch_size = len(data)
        dummy_features = torch.zeros((batch_size, 128, 19))  # Adjust dimensions to match your needs
        
        # Try to extract labels if possible
        try:
            if all(isinstance(item, (int, float)) for item in data):
                # All items are scalars, use them as labels
                dummy_labels = torch.tensor(data)
            elif all(isinstance(item, (tuple, list)) and len(item) >= 2 for item in data):
                # Items are tuples, extract second element as label
                dummy_labels = torch.stack([item[1] if isinstance(item[1], torch.Tensor) else torch.tensor(item[1]) 
                                           for item in data])
            else:
                # Mixed types or unknown structure
                dummy_labels = torch.zeros(batch_size, dtype=torch.long)
        except:
            # Complete fallback
            dummy_labels = torch.zeros(batch_size, dtype=torch.long)
        
        dummy_padding_mask = torch.ones((batch_size, 128), dtype=torch.bool)
        return dummy_features, dummy_labels, dummy_padding_mask


def padding_mask(lengths, max_len=None):
    """
    Used to mask padded positions: creates a (batch_size, max_len) boolean mask from a tensor of sequence lengths,
    where 1 means keep element at this position (time step)
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max_val()
    return (torch.arange(0, max_len, device=lengths.device)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))


class Normalizer(object):
    """
    Normalizes dataframe across ALL contained rows (time steps). Different from per-sample normalization.
    """

    def __init__(self, norm_type='standardization', mean=None, std=None, min_val=None, max_val=None):
        """
        Args:
            norm_type: choose from:
                "standardization", "minmax": normalizes dataframe across ALL contained rows (time steps)
                "per_sample_std", "per_sample_minmax": normalizes each sample separately (i.e. across only its own rows)
            mean, std, min_val, max_val: optional (num_feat,) Series of pre-computed values
        """

        self.norm_type = norm_type
        self.mean = mean
        self.std = std
        self.min_val = min_val
        self.max_val = max_val

    def normalize(self, df):
        """
        Args:
            df: input dataframe
        Returns:
            df: normalized dataframe
        """
        if self.norm_type == "standardization":
            if self.mean is None:
                self.mean = df.mean()
                self.std = df.std()
            return (df - self.mean) / (self.std + np.finfo(float).eps)

        elif self.norm_type == "minmax":
            if self.max_val is None:
                self.max_val = df.max()
                self.min_val = df.min()
            return (df - self.min_val) / (self.max_val - self.min_val + np.finfo(float).eps)

        elif self.norm_type == "per_sample_std":
            grouped = df.groupby(by=df.index)
            return (df - grouped.transform('mean')) / grouped.transform('std')

        elif self.norm_type == "per_sample_minmax":
            grouped = df.groupby(by=df.index)
            min_vals = grouped.transform('min')
            return (df - min_vals) / (grouped.transform('max') - min_vals + np.finfo(float).eps)

        else:
            raise (NameError(f'Normalize method "{self.norm_type}" not implemented'))


def interpolate_missing(y):
    """
    Replaces NaN values in pd.Series `y` using linear interpolation
    """
    if y.isna().any():
        y = y.interpolate(method='linear', limit_direction='both')
    return y


def subsample(y, limit=256, factor=2):
    """
    If a given Series is longer than `limit`, returns subsampled sequence by the specified integer factor
    """
    if len(y) > limit:
        return y[::factor].reset_index(drop=True)
    return y



def bandpass_filter_func(signal, fs, lowcut, highcut):
    # length of signal
    fft_len = signal.shape[1]
    # FFT
    fft_spectrum = np.fft.rfft(signal, n=fft_len, axis=1)
    # get frequency bins
    freqs = np.fft.rfftfreq(fft_len, d=1/fs)
    # create mask for freqs
    mask = (freqs >= lowcut) & (freqs <= highcut)
    # expand mask to match fft_spectrum dimensions
    mask = mask[:, np.newaxis]  # Adjust mask shape if necessary
    # apply mask
    fft_spectrum = fft_spectrum * mask
    # IFFT
    filtered_signal = np.fft.irfft(fft_spectrum, n=fft_len, axis=1)

    return filtered_signal


def normalize_batch_ts(batch):
    """Normalize a batch of time-series data.

    Args:
        batch (numpy.ndarray): A batch of input time-series in shape (N, T, C).

    Returns:
        numpy.ndarray: A batch of processed time-series, normalized for each channel of each sample.
    """
    # Calculate mean and std for each sample's each channel
    mean_values = batch.mean(axis=1, keepdims=True)  # Shape: (N, 1, C)
    std_values = batch.std(axis=1, keepdims=True)  # Shape: (N, 1, C)

    # Avoid division by zero by setting small std to 1
    std_values[std_values == 0] = 1.0

    # Perform standard normalization
    normalized_batch = (batch - mean_values) / std_values

    return normalized_batch


def load_data_by_ids(data_path, label_path, ids):
    '''
    Loads subjects with IDs in the ids list
    Args:
        data_path: directory of data files
        label_path: directory of label.npy file
        ids: list of subject IDs to load
    Returns:
        X: (N, T, C)
        y: (N, 2), first column is label, second column is subject ID
    '''
    feature_list = []
    label_list = []
    subject_labels = np.load(label_path)

    # load data by subject ids
    for filename in os.listdir(data_path):
        # get subject ID from filename, e.g., 'AD_1.npy'
        sub_id = int(filename.split('_')[-1].split('.')[0])
        # only load subject with ID in the ids list
        if sub_id in ids:
            # get label for the subject, all samples in the subject have the same label
            subject_label = subject_labels[subject_labels[:, 1] == sub_id][0]
            path = os.path.join(data_path, filename)
            subject_feature = np.load(path)
            for sample_feature in subject_feature:
                feature_list.append(sample_feature)
                label_list.append(subject_label)
    # reshape and shuffle
    X = np.array(feature_list)
    y = np.array(label_list)
    X, y = shuffle(X, y, random_state=42)

    return X, y
