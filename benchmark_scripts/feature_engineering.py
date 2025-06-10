import numpy as np
from sklearn.preprocessing import LabelEncoder

def build_features(df, level='subfamily', max_domains=50):
    """
    Builds a model-agnostic feature matrix (X) and target vector (y) from the protein data.
    This function encapsulates the feature engineering logic from the original ProteinDataset.

    Args:
        df (pd.DataFrame): The input DataFrame from data_loader.
        level (str): The classification level, 'subfamily' or 'family'.
        max_domains (int): The maximum number of domains to consider for order features.

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
                domain_scores[domain_idx] = np.log1p(score)
                ordered_domains.append(domain_idx)

        # Process separators
        separator_features = []
        for sep in row['Seperators']:
            _, start_pos, end_pos = sep
            start_norm = start_pos / row['Length']
            end_norm = end_pos / row['Length']
            length_norm = end_norm - start_norm
            separator_features.extend([start_norm, end_norm, length_norm])
        
        # Pad separator features
        separator_features = (separator_features + [0] * 60)[:60]

        # Domain order features
        order_features = np.zeros(max_domains)
        for i, domain_idx in enumerate(ordered_domains[:max_domains]):
            order_features[i] = domain_idx
        
        # Domain count feature
        domain_count = len(domains) / max_domains

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