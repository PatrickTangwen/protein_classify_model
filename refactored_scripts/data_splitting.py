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
    For each class in the test set, generate negative control proteins from other superfamilies.
    
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
    print("Generating negative control sets...")
    target_col = 'Family' if level == 'family' else 'Subfamily'

    # Get target class for each test index
    target_to_test_indices = defaultdict(list)
    for idx in test_indices:
        target_class = df.iloc[idx][target_col]
        target_to_test_indices[target_class].append(idx)
    
    if level == 'subfamily':
        # Extract family prefix (first three parts) from each subfamily
        def get_family_prefix(subfamily):
            return '.'.join(subfamily.split('.')[:3])
        
        # Create mapping from subfamily to family
        subfamily_to_family = {subfamily: get_family_prefix(subfamily) for subfamily in df['Subfamily'].unique()}
        
        # Create mapping from family to superfamily
        family_superfamily_map = {}
        for subfamily in df['Subfamily'].unique():
            family = subfamily_to_family[subfamily]
            if family in family_to_superfamily_map:
                family_superfamily_map[family] = family_to_superfamily_map[family]
    else:
        family_superfamily_map = family_to_superfamily_map
    
    print(f"Found {len(family_superfamily_map)} families with superfamily assignments")
    
    # Generate negative controls for each target class
    negative_control_indices = {}
    
    for target_class, class_test_indices in target_to_test_indices.items():
        if level == 'subfamily':
            family = subfamily_to_family[target_class]
        else:
            family = target_class
            
        target_superfamily = family_superfamily_map.get(family)
        
        # Determine how many negative controls we need
        n_test = len(class_test_indices)
        n_negative = max(n_test, 5)  # At least 5 negative controls
        
        # Find eligible proteins from other superfamilies
        eligible_indices = []
        
        for idx, row in df.iterrows():
            if idx in train_indices:  # Skip training proteins
                continue
                
            other_target_class = row[target_col]
            
            # Skip if same target class
            if other_target_class == target_class:
                continue
            
            if level == 'subfamily':
                other_family = subfamily_to_family[other_target_class]
            else:
                other_family = other_target_class
                
            # If target class has no superfamily, only use proteins from families with superfamily assignments
            if target_superfamily is None:
                if other_family in family_superfamily_map:
                    eligible_indices.append(idx)
            # If target has superfamily, use proteins from different superfamilies
            else:
                other_superfamily = family_superfamily_map.get(other_family)
                if other_superfamily is not None and other_superfamily != target_superfamily:
                    eligible_indices.append(idx)
        
        # Randomly select negative controls
        if len(eligible_indices) >= n_negative:
            negative_control_indices[target_class] = random.sample(eligible_indices, n_negative)
        else:
            # If not enough eligible proteins, use all available
            negative_control_indices[target_class] = eligible_indices
            print(f"Warning: Not enough negative controls for {target_col.lower()} {target_class}. "
                  f"Needed {n_negative}, found {len(eligible_indices)}.")
    
    return negative_control_indices, target_to_test_indices

def custom_split_dataset_with_negatives(df, family_to_superfamily_map, level='subfamily'):
    """
    Creates train and test splits with negative controls added to test set.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
        family_to_superfamily_map (dict): Mapping from family to superfamily.
        level (str): The classification level ('subfamily' or 'family').

    Returns:
        tuple: A tuple containing:
            - list: Indices for the training set.
            - list: Indices for the validation set (positives + negatives).
            - dict: Mapping of index to a boolean indicating if it's a negative control.
            - dict: Mapping of class to its positive and negative test indices.
    """
    print("\n=== Starting Data Preparation Process ===")
    # Get basic train/test split
    train_indices, test_indices = custom_split_dataset(df, level)
    
    # Generate negative controls
    negative_control_dict, target_to_test_indices = generate_negative_controls(
        df, test_indices, train_indices, family_to_superfamily_map, level
    )
    
    # Create combined test set with negative controls
    test_indices_with_negatives = test_indices.copy()
    is_negative_control = {idx: False for idx in test_indices}  # Track which are negative controls
    
    # Create mapping from target class to all its test indices (positive and negative)
    target_test_mapping = {}
    
    for target_class, class_test_indices in target_to_test_indices.items():
        negative_indices = negative_control_dict.get(target_class, [])
        
        # Add negative controls to test set
        for idx in negative_indices:
            if idx not in test_indices_with_negatives:  # Avoid duplicates
                test_indices_with_negatives.append(idx)
                is_negative_control[idx] = True
        
        # Store mapping of target class to all its test indices
        target_test_mapping[target_class] = {
            'positive': class_test_indices,
            'negative': negative_indices
        }
    
    print("=== Data Preparation Complete ===\n")
    
    return train_indices, test_indices_with_negatives, is_negative_control, target_test_mapping