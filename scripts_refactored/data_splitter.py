import random
import numpy as np
import pandas as pd
from collections import defaultdict

DEFAULT_SEED = 42

def set_seed(seed_value=DEFAULT_SEED):
    """Sets the seed for random, numpy for reproducible splits."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    # If using PyTorch for any splitting steps directly:
    # import torch
    # torch.manual_seed(seed_value)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed_all(seed_value)
    print(f"Global random seed set to {seed_value} for data splitting.")

def _perform_custom_split(df, group_column_name):
    """
    Implements custom splitting strategy based on group size, ensuring reproducibility.
    - 1 member: put in both train and test.
    - 2 members: split 1:1 (randomly).
    - >2 members: split 80:20 (randomly).
    """
    print(f"Starting custom data split by '{group_column_name}'...")
    train_indices = []
    test_indices = []

    for group_id, group_df in df.groupby(group_column_name):
        indices = group_df.index.tolist()
        n_samples = len(indices)
        
        shuffled_indices = list(indices)  # Create a mutable copy
        random.shuffle(shuffled_indices)  # Shuffle for reproducible random splits

        if n_samples == 1:
            train_indices.extend(shuffled_indices)
            test_indices.extend(shuffled_indices)
        elif n_samples == 2:
            train_indices.append(shuffled_indices[0])
            test_indices.append(shuffled_indices[1])
        else:
            n_train = int(0.8 * n_samples)
            # Ensure at least one sample in test if possible
            if n_train == n_samples and n_samples > 0:
                n_train = n_samples - 1
            
            current_train_indices = shuffled_indices[:n_train]
            current_test_indices = shuffled_indices[n_train:]

            # Handle edge case: if n_train makes test empty but train has all, move one to test
            if not current_test_indices and current_train_indices and len(current_train_indices) > 1:
                current_test_indices.append(current_train_indices.pop())
            # Handle edge case: if n_train makes train empty but test has all, move one to train
            if not current_train_indices and current_test_indices and len(current_test_indices) > 1:
                 current_train_indices.append(current_test_indices.pop(0))


            train_indices.extend(current_train_indices)
            test_indices.extend(current_test_indices)
            
    print(f"Custom data split for '{group_column_name}' complete. Train samples: {len(train_indices)}, Test samples: {len(test_indices)}")
    return train_indices, test_indices

def generate_negative_controls_for_groups(
    df,
    test_indices,
    train_indices,
    group_column_name,
    group_to_superfamily_map # Pre-computed map: group_id -> superfamily_id
):
    """
    For each group in the test set, generates negative control proteins from other superfamilies.
    """
    print(f"Generating negative control sets for '{group_column_name}'...")
    
    # Map each group_id in the test set to its original positive test indices
    group_to_original_test_indices = defaultdict(list)
    for idx in test_indices:
        group_id = df.loc[idx, group_column_name]
        group_to_original_test_indices[group_id].append(idx)

    num_groups_with_superfamily = sum(1 for sf in group_to_superfamily_map.values() if sf is not None and sf != 'Unknown')
    print(f"Found {num_groups_with_superfamily} groups with superfamily assignments in the provided map.")

    negative_controls_for_group = {} # Maps group_id -> list of its negative control indices

    for target_group, current_group_test_indices in group_to_original_test_indices.items():
        target_superfamily = group_to_superfamily_map.get(target_group) # Superfamily of the current group being processed
        
        n_test = len(current_group_test_indices)
        n_negative = max(n_test, 5)

        eligible_indices_for_neg_control = []
        for idx, row_data in df.iterrows():
            # Protein already in train set for the main task cannot be a negative control
            if idx in train_indices:
                continue

            other_group = row_data[group_column_name]
            
            # Proteins from the same group cannot be negative controls for this target_group
            if other_group == target_group:
                continue
            
            other_superfamily = group_to_superfamily_map.get(other_group)

            # Selection logic for negative controls:
            # If the target_group has no defined superfamily,
            # eligible negative controls must come from groups that *do* have a superfamily.
            if target_superfamily is None or target_superfamily == 'Unknown':
                if other_superfamily is not None and other_superfamily != 'Unknown':
                    eligible_indices_for_neg_control.append(idx)
            # If the target_group *has* a superfamily,
            # eligible negative controls must come from groups belonging to a *different* superfamily.
            else:
                if other_superfamily is not None and other_superfamily != 'Unknown' and other_superfamily != target_superfamily:
                    eligible_indices_for_neg_control.append(idx)
        
        # Remove duplicates that might have arisen
        eligible_indices_for_neg_control = list(set(eligible_indices_for_neg_control))
        
        # Ensure negative controls are not part of the positive test set for the current target_group
        eligible_indices_for_neg_control = [i for i in eligible_indices_for_neg_control if i not in current_group_test_indices]


        if len(eligible_indices_for_neg_control) >= n_negative:
            negative_controls_for_group[target_group] = random.sample(eligible_indices_for_neg_control, n_negative)
        else:
            negative_controls_for_group[target_group] = eligible_indices_for_neg_control
            if n_negative > 0 : # Only warn if we actually needed some
                print(f"Warning: Not enough distinct negative controls for group {target_group}. "
                      f"Needed {n_negative}, found {len(eligible_indices_for_neg_control)}.")
    
    print("Negative control set generation complete.")
    return negative_controls_for_group, group_to_original_test_indices

def split_dataset_with_negative_controls(
    df,
    group_column_name,
    group_to_superfamily_map, # Pre-computed: group_id (family/subfamily) -> superfamily_id
    seed=DEFAULT_SEED
):
    """
    Creates train and test splits with negative controls added to the test set.
    `group_to_superfamily_map` should map entities from `group_column_name` to their respective superfamilies.
    Returns:
    - train_indices: List of indices for training.
    - final_test_indices: List of indices for the combined test set (original test + all negative controls).
    - is_negative_control_marker: Dict {index: bool} indicating if an index in final_test_indices is a negative control.
    - detailed_group_test_mapping: Dict {group_id: {'positive': [indices], 'negative': [indices]}}.
    """
    if seed is not None:
        set_seed(seed)

    print(f"\n=== Starting Data Preparation Process for column '{group_column_name}' ===")
    
    # 1. Get basic train/test split
    train_indices, original_test_indices = _perform_custom_split(df, group_column_name)
    
    # 2. Generate negative controls based on the original_test_indices
    negative_control_dict, group_to_positive_test_indices = generate_negative_controls_for_groups(
        df, original_test_indices, train_indices, group_column_name, group_to_superfamily_map
    )
    
    # 3. Create combined test set and mappings
    final_test_indices = list(original_test_indices) # Start with positive test samples
    is_negative_control_marker = {idx: False for idx in original_test_indices} 
                                 # True if index is a negative control for *any* group

    detailed_group_test_mapping = {}

    # Initialize with positive test samples for each group
    for group_id, pos_indices in group_to_positive_test_indices.items():
        detailed_group_test_mapping[group_id] = {
            'positive': list(pos_indices), 
            'negative': [] 
        }

    all_added_negative_indices = []
    for group_id, neg_indices_for_this_group in negative_control_dict.items():
        if group_id not in detailed_group_test_mapping: 
             detailed_group_test_mapping[group_id] = {'positive': [], 'negative': []} 
        
        detailed_group_test_mapping[group_id]['negative'].extend(neg_indices_for_this_group)
        
        for neg_idx in neg_indices_for_this_group:
            all_added_negative_indices.append(neg_idx)
            is_negative_control_marker[neg_idx] = True 

    unique_negative_indices = sorted(list(set(all_added_negative_indices)))
    for neg_idx in unique_negative_indices:
        if neg_idx not in final_test_indices:
            final_test_indices.append(neg_idx)
            
    all_groups_in_df = df[group_column_name].unique()
    for group_id in all_groups_in_df:
        if group_id not in detailed_group_test_mapping:
            detailed_group_test_mapping[group_id] = {
                'positive': group_to_positive_test_indices.get(group_id, []),
                'negative': negative_control_dict.get(group_id, [])     
            }
        if 'positive' not in detailed_group_test_mapping[group_id]:
             detailed_group_test_mapping[group_id]['positive'] = group_to_positive_test_indices.get(group_id, [])
        if 'negative' not in detailed_group_test_mapping[group_id]:
             detailed_group_test_mapping[group_id]['negative'] = negative_control_dict.get(group_id, [])

    train_indices.sort()
    final_test_indices.sort()

    print(f"Total train indices: {len(train_indices)}")
    print(f"Total test indices (including all negative controls): {len(final_test_indices)}")
    num_marked_negative = sum(1 for v in is_negative_control_marker.values() if v)
    print(f"Number of unique proteins marked as negative controls: {num_marked_negative}")
    print(f"Detailed group test mapping created for {len(detailed_group_test_mapping)} groups.")
    print(f"=== Data Preparation for column '{group_column_name}' Complete ===\n")
    
    return train_indices, final_test_indices, is_negative_control_marker, detailed_group_test_mapping