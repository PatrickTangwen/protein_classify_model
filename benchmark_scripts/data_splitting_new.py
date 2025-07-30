import random
from collections import defaultdict

def custom_split_dataset(df, level='subfamily'):
    """
    Implements the custom data splitting strategy based on class size.
    - 1 member: put in both train and test sets.
    - 2 members: split 1:1 for train/test.
    - >2 members: split 80:20 for train/test.

    Args:
        df (pd.DataFrame): The input DataFrame.
        level (str): The classification level ('subfamily' or 'family').

    Returns:
        tuple: A tuple containing two lists of indices: train_indices and test_indices.
    """
    print("Starting data splitting process...")
    target_col = 'Family' if level == 'family' else 'Subfamily'
    train_indices = []
    test_indices = []

    # Group by target column to handle each case
    target_counts = {}
    for target_class, group in df.groupby(target_col):
        indices = group.index.tolist()
        n_samples = len(indices)
        target_counts[target_class] = n_samples
        
        if n_samples == 1:
            # Case 1: Single member goes to both sets
            train_indices.extend(indices)
            test_indices.extend(indices)
        elif n_samples == 2:
            # Case 2: Split 1:1
            train_indices.append(indices[0])
            test_indices.append(indices[1])
        else:
            # Case 3: Split 80:20
            n_train = int(0.8 * n_samples)
            train_indices.extend(indices[:n_train])
            test_indices.extend(indices[n_train:])
            
    print(f"Data splitting complete")
    return train_indices, test_indices

def generate_negative_controls(df, test_indices, train_indices, family_to_superfamily_map, level='subfamily'):
    """
    For each class in the test set, generate negative control proteins from other families
    within the same superfamily.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
        test_indices (list): List of indices for the test set.
        train_indices (list): List of indices for the train set.
        family_to_superfamily_map (dict): Mapping from family to superfamily.
        level (str): The classification level ('subfamily' or 'family').

    Returns:
        tuple: A tuple containing:
            - dict: Mapping from class to its list of negative control indices.
            - dict: Mapping from class to its list of positive test indices.
    """
    print("Generating negative control sets (new method)...")
    target_col = 'Family' if level == 'family' else 'Subfamily'

    # Map subfamilies to families if needed
    if level == 'subfamily':
        subfamily_to_family = {subf: '.'.join(subf.split('.')[:3]) for subf in df['Subfamily'].unique()}
    else:
        subfamily_to_family = None

    # Filter test indices to only include those with superfamily assignments
    valid_test_indices = []
    for idx in test_indices:
        target_class = df.iloc[idx][target_col]
        family = subfamily_to_family[target_class] if level == 'subfamily' else target_class
        if family in family_to_superfamily_map:
            valid_test_indices.append(idx)
        else:
            print(f"Excluding {target_class} from test set (no superfamily assignment).")
    
    # Get target class for each valid test index
    target_to_test_indices = defaultdict(list)
    for idx in valid_test_indices:
        target_class = df.iloc[idx][target_col]
        target_to_test_indices[target_class].append(idx)

    # Generate negative controls for each target class
    negative_control_indices = {}
    
    for target_class, class_test_indices in target_to_test_indices.items():
        family = subfamily_to_family[target_class] if level == 'subfamily' else target_class
        target_superfamily = family_to_superfamily_map.get(family)
        
        if not target_superfamily:
            continue
            
        n_test = len(class_test_indices)
        n_negative = max(n_test, 5)

        # Find eligible proteins from different families within the same superfamily
        eligible_indices = []
        for idx, row in df.iterrows():
            if idx in train_indices or idx in class_test_indices:
                continue
                
            other_target_class = row[target_col]
            other_family = subfamily_to_family[other_target_class] if level == 'subfamily' else other_target_class
            
            if other_family == family:
                continue

            other_superfamily = family_to_superfamily_map.get(other_family)
            if other_superfamily == target_superfamily:
                eligible_indices.append(idx)
        
        # Randomly select negative controls
        if len(eligible_indices) >= n_negative:
            negative_control_indices[target_class] = random.sample(eligible_indices, n_negative)
        else:
            negative_control_indices[target_class] = eligible_indices
            if eligible_indices:
                 print(f"Warning: Not enough negative controls for {target_col.lower()} {target_class}. "
                      f"Needed {n_negative}, found {len(eligible_indices)}.")

    return negative_control_indices, target_to_test_indices

def custom_split_dataset_with_negatives(df, family_to_superfamily_map, level='subfamily'):
    """
    Creates train and test splits with negative controls added to test set.
    """
    print("\n=== Starting Data Preparation Process (New Method) ===")
    train_indices, test_indices = custom_split_dataset(df, level)
    
    negative_control_dict, target_to_test_indices = generate_negative_controls(
        df, test_indices, train_indices, family_to_superfamily_map, level
    )
    
    # The new test set only contains positives from families with superfamily mapping
    final_test_indices = [idx for indices in target_to_test_indices.values() for idx in indices]
    
    test_indices_with_negatives = final_test_indices.copy()
    is_negative_control = {idx: False for idx in final_test_indices}
    
    target_test_mapping = {}
    
    all_negative_indices = set()
    for target_class, class_test_indices in target_to_test_indices.items():
        negative_indices = negative_control_dict.get(target_class, [])
        all_negative_indices.update(negative_indices)
        
        target_test_mapping[target_class] = {
            'positive': class_test_indices,
            'negative': negative_indices
        }

    for idx in all_negative_indices:
        if idx not in test_indices_with_negatives:
            test_indices_with_negatives.append(idx)
            is_negative_control[idx] = True

    print("=== Data Preparation Complete ===\n")
    
    return train_indices, test_indices_with_negatives, is_negative_control, target_test_mapping