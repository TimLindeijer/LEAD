import numpy as np
import torch

def filter_by_health_status(data_set, health_filter='all', label_mapping='0,1,2'):
    """
    Filter dataset by health status (healthy control, MCI, or dementia)
    
    Args:
        data_set: Dataset object (must expose X and y attributes)
        health_filter: One of 'all', 'hc' (healthy controls), 'mci', or 'dementia'
        label_mapping: Comma-separated mapping of label values [hc,mci,dementia]
        
    Returns:
        Modified data_set with filtered samples
    """
    if health_filter.lower() == 'all':
        # No filtering needed
        return data_set
        
    # Parse the label mapping
    try:
        hc_label, mci_label, dementia_label = [int(x) for x in label_mapping.split(',')]
    except ValueError:
        print(f"Warning: Could not parse label_mapping '{label_mapping}'. Using default 0,1,2")
        hc_label, mci_label, dementia_label = 0, 1, 2
    
    # Determine which label to filter by
    filter_label = None
    if health_filter.lower() == 'hc':
        filter_label = hc_label
        status_name = "Healthy Controls"
    elif health_filter.lower() == 'mci':
        filter_label = mci_label
        status_name = "MCI"
    elif health_filter.lower() == 'dementia':
        filter_label = dementia_label
        status_name = "Dementia"
    else:
        print(f"Warning: Unknown health_filter '{health_filter}'. No filtering applied.")
        return data_set
    
    # Access data from the dataset object
    try:
        # Check if dataset has the required attributes
        if not hasattr(data_set, 'X') or not hasattr(data_set, 'y'):
            print(f"Warning: Dataset doesn't have required X and y attributes. Cannot filter.")
            return data_set

        # Backup original __getitem__ method
        original_getitem = data_set.__getitem__
        dataset_length = len(data_set)
        
        # Get current data
        X = data_set.X
        y = data_set.y
        
        # Safe check on types
        if isinstance(y, list):
            y = np.array(y)
        elif isinstance(y, torch.Tensor):
            y = y.numpy()
            
        # Debugging information
        print(f"Dataset filtering info:")
        print(f"X shape: {X.shape if hasattr(X, 'shape') else 'unknown'}")
        print(f"y shape: {y.shape if hasattr(y, 'shape') else 'unknown'}")
        print(f"y type: {type(y)}")
        
        # Determine the column with health status information
        health_col = 0
        if y.ndim > 1 and y.shape[1] > 1:
            # Try to intelligently guess which column contains health status
            # Common patterns in health datasets:
            # - First column (index 0): diagnostic label (HC=0, MCI=1, Dementia=2)
            # - Second column (index 1): subject ID
            
            # Look at unique values in each column to guess
            for col in range(y.shape[1]):
                unique_vals = np.unique(y[:, col])
                if len(unique_vals) <= 10:  # Health status typically has few categories
                    # Check if filter_label is in this column
                    if filter_label in unique_vals:
                        health_col = col
                        print(f"Detected health status in column {col}")
                        break
        
        # Create filter mask
        if y.ndim > 1:
            mask = (y[:, health_col] == filter_label)
        else:
            mask = (y == filter_label)
        
        # Count filtered samples
        filtered_count = np.sum(mask)
        total_count = len(y)
        
        if filtered_count == 0:
            print(f"Warning: No samples found with health status '{health_filter}' (label {filter_label})!")
            print(f"Available labels: {np.unique(y[:, health_col] if y.ndim > 1 else y)}")
            return data_set
        
        print(f"Filtered to {status_name} samples: {filtered_count}/{total_count} samples retained ({filtered_count/total_count*100:.1f}%)")
        
        # Apply filter to dataset
        filtered_X = X[mask]
        filtered_y = y[mask]
        
        # Update dataset with filtered data
        data_set.X = filtered_X
        data_set.y = filtered_y
        
        # Create filtered indices list for faster lookup
        filtered_indices = np.where(mask)[0]
        
        # Create a mapping from new indices to old indices
        index_map = {new_idx: old_idx for new_idx, old_idx in enumerate(filtered_indices)}
        
        # Override __getitem__ to handle the filtered data properly
        def new_getitem(self, idx):
            if hasattr(self, 'X_org') and hasattr(self, 'y_org'):
                # If using transformed data
                x = self.X[idx]
                y = self.y[idx]
            else:
                # Standard access
                x = self.X[idx]
                y = self.y[idx]
                
            # Call any additional processing the original __getitem__ might do
            if callable(original_getitem):
                try:
                    # Try with original index mapping
                    if idx in index_map:
                        original_idx = index_map[idx]
                        return original_getitem(original_idx)
                except:
                    # Fallback to default format
                    pass
            
            # Otherwise return standard format
            return x, y
        
        # Override __len__ for the correct count
        def new_len(self):
            return len(self.X)
        
        # Bind new methods to the dataset
        data_set.__getitem__ = new_getitem.__get__(data_set)
        data_set.__len__ = new_len.__get__(data_set)
        
        print(f"Updated dataset: {len(data_set)} samples")
        return data_set
        
    except Exception as e:
        print(f"Error applying health filter: {e}")
        import traceback
        traceback.print_exc()
        return data_set  # Return unfiltered dataset in case of errors