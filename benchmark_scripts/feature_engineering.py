import numpy as np
from sklearn.preprocessing import LabelEncoder

def build_features(df, level='subfamily', max_domains=50, max_separators=20, evalue_threshold=1e-300):
    """
    Builds a model-agnostic feature matrix (X) and target vector (y) from the protein data.
    This function encapsulates the feature engineering logic from the original ProteinDataset.

    Args:
        df (pd.DataFrame): The input DataFrame from data_loader.
        level (str): The classification level, 'subfamily' or 'family'.
        max_domains (int): The maximum number of domains to consider for order features.
        max_separators (int): The maximum number of separators to consider for separator features.
        evalue_threshold (float): The threshold for e-values to consider for domain scores.
    Returns:
        tuple: A tuple containing:
            - np.ndarray: The feature matrix (X).
            - np.ndarray: The encoded labels (y).
            - LabelEncoder: The fitted label encoder.
            - dict: The domain vocabulary.
            - tuple: The mean and standard deviation used for feature scaling.
    """
    target_col = 'Family' if level == 'family' else 'Subfamily'

    # 1. Label Encoding
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df[target_col])

    # 2. Build domain vocabulary
    all_domains = set(domain[0] for domains_list in df['Domains'] for domain in domains_list)
    domain_vocab = {acc: idx for idx, acc in enumerate(sorted(all_domains))}

    # 3. Feature extraction
    features_list = []
    for _, row in df.iterrows():
        domains = sorted(row['Domains'], key=lambda x: x[1])
        
        # Initialize feature arrays
        domain_presence = np.zeros(len(domain_vocab))
        domain_positions = np.zeros((len(domain_vocab), 2))
        domain_scores = np.zeros(len(domain_vocab))
        ordered_domains = []

        # Process domains
        for domain in domains:
            domain_acc, start_pos, end_pos, score = domain
            if domain_acc in domain_vocab:
                domain_idx = domain_vocab[domain_acc]
                domain_presence[domain_idx] = 1
                domain_positions[domain_idx] = [start_pos / row['Length'], end_pos / row['Length']]
                # For e-values: lower is better, so we need to invert the score
                # Clamp very small e-values to avoid numerical issues
                clamped_score = max(score, evalue_threshold) # set the threshold to 1e-300
                # Use negative log to convert e-values to scores (higher is better)
                # Example: score = 1e-30, max(1e-30, 1e-300) = 1e-30, -np.log10(1e-30 + 1e-300) = 30
                domain_scores[domain_idx] = -np.log10(clamped_score + 1e-300)
                ordered_domains.append(domain_idx)

        # Process separators
        separator_features = []
        separators_to_process = row['Seperators'][:max_separators] 
        for sep in separators_to_process:
            _, start_pos, end_pos = sep
            length = end_pos - start_pos
            separator_features.extend([start_pos, end_pos, length])
        

        # Pad to FIXED size
        separator_feature_size = max_separators * 3 
        separator_features = (separator_features + [0] * separator_feature_size)[:separator_feature_size]

        # Domain order features
        order_features = np.zeros(max_domains)
        for i, domain_idx in enumerate(ordered_domains[:max_domains]):
            order_features[i] = domain_idx
        
        # Domain count feature
        domain_count = len(domains)

        # Combine all features into a single vector
        final_features = np.concatenate([
            domain_presence,
            domain_positions.flatten(),
            domain_scores,
            np.array(separator_features),
            order_features,
            [domain_count]
        ])
        features_list.append(final_features)

    X = np.array(features_list, dtype=np.float32)

    # 4. Feature normalization
    feature_mean = X.mean(axis=0)
    feature_std = X.std(axis=0)
    feature_std[feature_std == 0] = 1  # Avoid division by zero
    X = (X - feature_mean) / feature_std
    
    return X, y, label_encoder, domain_vocab, (feature_mean, feature_std) 