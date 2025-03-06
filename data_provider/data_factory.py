from data_provider.data_loader import MultiDatasetsLoader
from data_provider.data_loader import SingleDatasetLoader
from data_provider.health_filter import filter_by_health_status
from data_provider.subject_counter import count_subjects_by_health_status

from data_provider.uea import collate_fn
from torch.utils.data import DataLoader
from utils.tools import CustomGroupSampler

# data type dict to loader mapping
data_type_dict = {
    # loading single dataset
    'SingleDataset': SingleDatasetLoader,

    # loading multiple datasets, concatenating them
    'MultiDatasets': MultiDatasetsLoader,  # datasets folder names presented in args.data_folder_list
}


def data_provider(args, flag):
    Data = data_type_dict[args.data]

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        if args.task_name == 'supervised'\
                or args.task_name == 'pretrain_lead' \
                or args.task_name == 'pretrain_moco' \
                or args.task_name == 'pretrain_ts2vec' \
                or args.task_name == 'pretrain_biot' \
                or args.task_name == 'pretrain_eeg2rep' \
                or args.task_name == 'finetune' \
                or args.task_name == 'diffusion':  # Added diffusion here
            batch_size = args.batch_size
        else:
            batch_size = 1  # bsz=1 for evaluation
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size  # bsz for train and valid

    if args.task_name == 'supervised' \
            or args.task_name == 'pretrain_moco' \
            or args.task_name == 'pretrain_ts2vec' \
            or args.task_name == 'pretrain_biot' \
            or args.task_name == 'pretrain_eeg2rep' \
            or args.task_name == 'finetune' \
            or args.task_name == 'diffusion':  # Added diffusion here
        drop_last = False
        
        # Load the dataset
        try:
            data_set = Data(
                root_path=args.root_path,
                args=args,
                flag=flag,
            )
            
            # Print some debug info about the dataset
            print(f"Dataset loaded: {type(data_set)}")
            if hasattr(data_set, 'X'):
                print(f"X shape: {data_set.X.shape if hasattr(data_set.X, 'shape') else 'unknown'}")
            if hasattr(data_set, 'y'):
                print(f"y shape: {data_set.y.shape if hasattr(data_set.y, 'shape') else 'unknown'}")
                
            # Check __getitem__ implementation  
            if len(data_set) > 0:
                try:
                    first_item = data_set[0]
                    print(f"First item type: {type(first_item)}")
                    if isinstance(first_item, tuple) and len(first_item) >= 2:
                        print(f"Item structure looks good: {first_item[0].shape}, {first_item[1].shape if hasattr(first_item[1], 'shape') else first_item[1]}")
                    else:
                        print(f"WARNING: Item is not a tuple with (feature, label) structure: {first_item}")
                except Exception as e:
                    print(f"Error testing __getitem__: {e}")
            
        except Exception as e:
            print(f"Error loading dataset: {e}")
            import traceback
            traceback.print_exc()
            raise
        
        # Count subjects by health status before filtering data
        if hasattr(args, 'health_filter') and hasattr(args, 'subject_conditional') and args.subject_conditional:
            health_filter = getattr(args, 'health_filter', 'all')
            label_mapping = getattr(args, 'label_mapping', '0,1,2')
            
            # Before filtering, count how many subjects will be in the filtered data
            if flag.lower() == 'train' or flag.lower() == 'training':
                # Only update subject count based on training data
                try:
                    subject_count = count_subjects_by_health_status(
                        args.root_path,
                        health_filter,
                        label_mapping
                    )
                    
                    # Update the args with accurate subject count
                    original_num_subjects = getattr(args, 'num_subjects', -1)
                    if original_num_subjects != subject_count:
                        print(f"Updating number of subjects from {original_num_subjects} to {subject_count}")
                        args.num_subjects = subject_count
                        
                        # Store original value for reference
                        if not hasattr(args, 'original_num_subjects'):
                            args.original_num_subjects = original_num_subjects
                except Exception as e:
                    print(f"Error counting subjects: {e}")
                    import traceback
                    traceback.print_exc()
        
        # Apply health status filtering if requested
        if hasattr(args, 'health_filter') and args.health_filter.lower() != 'all':
            # Apply the filtering using our utility
            try:
                label_mapping = getattr(args, 'label_mapping', '0,1,2')
                data_set = filter_by_health_status(data_set, args.health_filter, label_mapping)
                
                # Verify the dataset after filtering
                print(f"Dataset after filtering: {len(data_set)} samples")
                
                # Test __getitem__ after filtering
                if len(data_set) > 0:
                    try:
                        first_item = data_set[0]
                        print(f"First item after filtering: {type(first_item)}")
                        if isinstance(first_item, tuple) and len(first_item) >= 2:
                            print(f"Filtered item looks good: {first_item[0].shape}, {first_item[1].shape if hasattr(first_item[1], 'shape') else first_item[1]}")
                        else:
                            print(f"WARNING: Filtered item is not a tuple: {first_item}")
                    except Exception as e:
                        print(f"Error testing filtered __getitem__: {e}")
            except Exception as e:
                print(f"Error during health filtering: {e}")
                import traceback
                traceback.print_exc()

        # Create data loader with the fixed collate function
        try:
            data_loader = DataLoader(
                data_set,
                batch_size=batch_size,
                shuffle=shuffle_flag,
                num_workers=args.num_workers,
                drop_last=drop_last,
                collate_fn=lambda x: collate_fn(x, max_len=args.seq_len)
            )
            
            # Verify the data loader
            print(f"Created DataLoader with {len(data_loader)} batches")
            
            return data_set, data_loader
            
        except Exception as e:
            print(f"Error creating DataLoader: {e}")
            import traceback
            traceback.print_exc()
            raise