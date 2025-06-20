import pandas as pd
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import ast
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import WeightedRandomSampler
from collections import defaultdict
from scipy.special import softmax
import os
import sys
from datetime import datetime
from tabulate import tabulate


# Create model_results_family directory outside scripts folder if it doesn't exist
results_dir = os.path.join('..', 'neural_network','model_results_family')
try:
    os.makedirs(results_dir, exist_ok=True)
    print(f"Results directory created/verified at: {results_dir}")
except Exception as e:
    print(f"Warning: Could not create results directory: {e}")
    results_dir = '.'  # Fallback to current directory
    print(f"Using current directory for results instead: {results_dir}")

# Set up logging to capture terminal output
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # Keep timestamp for reference
log_file = os.path.join(results_dir, 'training_log_family.txt')

# Load superfamily data
try:
    superfamily_data_path = os.path.join('..', 'data_source', 'fam2supefamily.csv')
    superfamily_df = pd.read_csv(superfamily_data_path)
    # Clean up whitespace in column names
    superfamily_df.columns = [col.strip() for col in superfamily_df.columns]
    # Create a mapping from family to superfamily
    family_to_superfamily = dict(zip(superfamily_df['family'].str.strip(), superfamily_df['label'].str.strip()))
except Exception as e:
    print(f"Warning: Could not load superfamily data: {e}")
    family_to_superfamily = {}

class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w', encoding='utf-8')
        self.silent_mode = False  # Flag to control when to stop terminal output
        
        # For classification report
        self.classification_report_file = None
        self.classification_mode = False

    def write(self, message):
        # Always write to log file or classification report file
        if self.classification_mode and self.classification_report_file:
            self.classification_report_file.write(message)
            self.classification_report_file.flush()
        else:
            self.log.write(message)
            self.log.flush()
        
        # Check if we should enter silent mode
        if "=== Detailed Family Classification Report ===" in message:
            self.silent_mode = True
            # Switch to classification report file
            self.classification_mode = True
            self.classification_report_file = open(os.path.join(results_dir, 'classification_report_family.txt'), 'w', encoding='utf-8')
            self.classification_report_file.write(message)
            self.classification_report_file.flush()
        
        # Write to terminal only if not in silent mode
        if not self.silent_mode:
            self.terminal.write(message)
            self.terminal.flush()

    def flush(self):
        if self.classification_mode and self.classification_report_file:
            self.classification_report_file.flush()
        else:
            self.log.flush()
            
        if not self.silent_mode:
            self.terminal.flush()

    def close(self):
        self.log.close()
        if self.classification_report_file:
            self.classification_report_file.close()

# Start logging
logger = Logger(log_file)
sys.stdout = logger

print(f"Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Results will be saved to: {results_dir}")
print("="*80)

# Load data
data_path = os.path.join('..', 'data_source', 'data_new.csv')
df = pd.read_csv(data_path)

# Extract family from subfamily for family-level classification
df['Family'] = df['Subfamily'].apply(lambda x: '.'.join(x.split('.')[:3]))


class ProteinDataset(Dataset):
    def __init__(self, df, max_domains=50):
        self.max_domains = max_domains
        
        # Convert string representations of lists to actual lists
        df['Domains'] = df['Domains'].apply(ast.literal_eval)
        df['Seperators'] = df['Seperators'].apply(ast.literal_eval)
        
        # Create label encoder for families (not subfamilies)
        self.label_encoder = LabelEncoder()
        encoded_labels = self.label_encoder.fit_transform(df['Family'])
        
        # Process features
        self.features = []
        self.labels = []
        
        # Get unique domain accessions for one-hot encoding
        all_domains = set()
        for row in df['Domains']:
            for domain in row:
                domain_acc = domain[0]  # Domain accession (e.g., CDD288812)
                all_domains.add(domain_acc)
        self.domain_vocab = {acc: idx for idx, acc in enumerate(sorted(all_domains))}
        
        # Initialize features as a list
        features_list = []
        
        for _, row in df.iterrows():
            # Sort domains by start position to maintain order
            domains = sorted(row['Domains'], key=lambda x: x[1])
            
            # Initialize features
            domain_features = []
            
            # 1. Domain presence and position features
            domain_presence = np.zeros(len(self.domain_vocab))
            domain_positions = np.zeros((len(self.domain_vocab), 2))  # start and end positions
            domain_scores = np.zeros(len(self.domain_vocab))
            
            # 2. Domain order features
            ordered_domains = []
            
            # 3. Domain transition features
            transitions = []
            
            # Process each domain
            prev_domain = None
            for i, domain in enumerate(domains):
                domain_acc = domain[0]  # Domain accession
                domain_idx = self.domain_vocab[domain_acc]
                
                # Update domain presence
                domain_presence[domain_idx] = 1
                
                # Update position features (normalized by protein length)
                domain_positions[domain_idx] = [
                    domain[1] / row['Length'],
                    domain[2] / row['Length']
                ]
                
                # Update domain scores (normalized)
                domain_scores[domain_idx] = np.log1p(domain[3])  # Log transform scores
                
                # Add to ordered domains
                ordered_domains.append(domain_idx)
                
                # Add transition if not first domain
                if prev_domain is not None:
                    transition = (prev_domain, domain_idx)
                    transitions.append(transition)
                prev_domain = domain_idx
            
            # 4. Process separators
            separator_features = []
            for sep in row['Seperators']:
                # Normalize positions by protein length
                start_norm = sep[1] / row['Length']
                end_norm = sep[2] / row['Length']
                length_norm = (end_norm - start_norm)
                separator_features.extend([start_norm, end_norm, length_norm])
            
            # Pad separator features
            separator_features = (separator_features + [0] * 60)[:60]  # Max 20 separators * 3 features
            
            # 5. Create final feature vector
            feature_vector = np.concatenate([
                domain_presence,  # Domain presence
                domain_positions.flatten(),  # Domain positions
                domain_scores,  # Domain scores
                np.array(separator_features)  # Separator features
            ])
            
            # Add order features
            order_features = np.zeros(self.max_domains)
            for i, domain_idx in enumerate(ordered_domains[:self.max_domains]):
                order_features[i] = domain_idx
            
            # Add domain count feature
            domain_count = len(domains) / self.max_domains  # Normalized domain count
            
            # Combine all features
            final_features = np.concatenate([
                feature_vector, 
                order_features,
                [domain_count]
            ])
            
            # Append to features_list
            features_list.append(final_features)
        
        # Convert list to numpy array first, then to tensor
        self.features = torch.FloatTensor(np.array(features_list))
        self.labels = torch.LongTensor(encoded_labels)
        
        # Normalize features
        self.feature_mean = self.features.mean(dim=0)
        self.feature_std = self.features.std(dim=0)
        self.feature_std[self.feature_std == 0] = 1  # Avoid division by zero
        self.features = (self.features - self.feature_mean) / self.feature_std
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            'features': self.features[idx],
            'label': self.labels[idx]
        }

class ImprovedProteinClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dims=[512, 256, 128]):
        super(ImprovedProteinClassifier, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Create hidden layers
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.4)
            ])
            prev_dim = hidden_dim
        
        self.feature_layers = nn.Sequential(*layers)
        
        # Final classification layer
        self.classifier = nn.Linear(hidden_dims[-1], num_classes)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        features = self.feature_layers(x)
        output = self.classifier(features)
        return output

def analyze_misclassification_type(true_family, pred_family):
    """
    For family-level classification, we can analyze superfamily-level errors
    Returns: 'same_superfamily' or 'different_superfamily'
    """
    true_superfamily = family_to_superfamily.get(true_family, 'Unknown')
    pred_superfamily = family_to_superfamily.get(pred_family, 'Unknown')
    
    if true_superfamily != 'Unknown' and pred_superfamily != 'Unknown' and true_superfamily == pred_superfamily:
        return 'same_superfamily'
    return 'different_superfamily'

def custom_split_dataset(df):
    """
    Implements custom splitting strategy based on family size:
    - 1 member: put in both train and test
    - 2 members: split 1:1
    - >2 members: split 80:20
    """
    print("Starting data splitting process...")
    train_indices = []
    test_indices = []
    
    # Group by family to handle each case
    family_counts = {}
    for family, group in df.groupby('Family'):
        indices = group.index.tolist()
        n_samples = len(indices)
        family_counts[family] = n_samples
        
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

def generate_negative_controls(df, test_indices, train_indices):
    """
    For each family in the test set, generate negative control proteins from other superfamilies.
    
    Returns:
    - negative_control_indices: Dictionary mapping family to list of negative control indices
    - family_to_test_indices: Dictionary mapping family to its test indices
    """
    print("Generating negative control sets...")
    # Get family for each test index
    family_to_test_indices = defaultdict(list)
    for idx in test_indices:
        family = df.iloc[idx]['Family']
        family_to_test_indices[family].append(idx)
    
    print(f"Found {len(family_to_superfamily)} families with superfamily assignments")
    
    # Generate negative controls for each family
    negative_control_indices = {}
    
    for family, family_test_indices in family_to_test_indices.items():
        target_superfamily = family_to_superfamily.get(family)
        
        # Determine how many negative controls we need
        n_test = len(family_test_indices)
        n_negative = max(n_test, 5)  # At least 5 negative controls
        
        # Find eligible proteins from other superfamilies
        eligible_indices = []
        
        for idx, row in df.iterrows():
            if idx in train_indices:  # Skip training proteins
                continue
                
            other_family = row['Family']
            
            # Skip if same family
            if other_family == family:
                continue
                
            # If target family has no superfamily, only use proteins from families with superfamily assignments
            if target_superfamily is None:
                if other_family in family_to_superfamily:
                    eligible_indices.append(idx)
            # If target has superfamily, use proteins from different superfamilies
            else:
                other_superfamily = family_to_superfamily.get(other_family)
                if other_superfamily is not None and other_superfamily != target_superfamily:
                    eligible_indices.append(idx)
        
        # Randomly select negative controls
        if len(eligible_indices) >= n_negative:
            negative_control_indices[family] = random.sample(eligible_indices, n_negative)
        else:
            # If not enough eligible proteins, use all available
            negative_control_indices[family] = eligible_indices
            print(f"Warning: Not enough negative controls for family {family}. "
                  f"Needed {n_negative}, found {len(eligible_indices)}.")
    
    return negative_control_indices, family_to_test_indices

def custom_split_dataset_with_negatives(df):
    """
    Creates train and test splits with negative controls added to test set.
    
    Returns:
    - train_indices: List of indices for training
    - test_indices_with_negatives: List of indices for testing (includes original test + negative controls)
    - is_negative_control: Dictionary mapping test index to boolean (True if negative control)
    - family_test_mapping: Dictionary mapping family to its test indices (both positive and negative)
    """
    print("\n=== Starting Data Preparation Process ===")
    # Get basic train/test split
    train_indices, test_indices = custom_split_dataset(df)
    
    # Generate negative controls
    negative_control_dict, family_to_test_indices = generate_negative_controls(df, test_indices, train_indices)
    
    # Create combined test set with negative controls
    test_indices_with_negatives = test_indices.copy()
    is_negative_control = {idx: False for idx in test_indices}  # Track which are negative controls
    
    # Create mapping from family to all its test indices (positive and negative)
    family_test_mapping = {}
    
    for family, family_test_indices in family_to_test_indices.items():
        negative_indices = negative_control_dict.get(family, [])
        
        # Add negative controls to test set
        for idx in negative_indices:
            if idx not in test_indices_with_negatives:  # Avoid duplicates
                test_indices_with_negatives.append(idx)
                is_negative_control[idx] = True
        
        # Store mapping of family to all its test indices
        family_test_mapping[family] = {
            'positive': family_test_indices,
            'negative': negative_indices
        }
    
    print("=== Data Preparation Complete ===\n")
    
    return train_indices, test_indices_with_negatives, is_negative_control, family_test_mapping

def evaluate_model_detailed(model, data_loader, dataset, device, original_df, train_indices, is_negative_control=None, family_test_mapping=None):
    model.eval()
    predictions = []
    true_labels = []
    confidences = []
    protein_ids = []
    family_metrics = defaultdict(lambda: {
        'train_count': 0,
        'test_count': 0,  # Count of original test samples
        'correct': 0,    # Correct predictions on original test samples
        'size': 0,  # Total size of family
        'misclassified': [],
        'same_superfamily_errors': 0,
        'different_superfamily_errors': 0,
        # New metrics for binary classification (family vs others, including negative controls)
        'TP': 0,  # True Positives
        'FP': 0,  # False Positives
        'TN': 0,  # True Negatives
        'FN': 0,  # False Negatives
        'negative_count': 0  # Number of negative controls associated with this family's test
    })
    
    # Calculate total size of each family
    family_counts = original_df['Family'].value_counts()
    for family, count in family_counts.items():
        family_metrics[family]['size'] = count
    
    # Count training samples
    for idx in train_indices:
        family = original_df.iloc[idx]['Family']
        family_metrics[family]['train_count'] += 1
    
    # Count negative controls if provided
    if family_test_mapping:
        for family, mapping in family_test_mapping.items():
            family_metrics[family]['negative_count'] = len(mapping['negative'])
    
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            features = batch['features'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(features)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, dim=1)
            
            predictions.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
            confidences.extend(confidence.cpu().numpy())
            
            if hasattr(data_loader.dataset, 'indices'):
                batch_indices = data_loader.dataset.indices[i*data_loader.batch_size:
                                                          (i+1)*data_loader.batch_size]
            else:
                batch_indices = list(range(i*data_loader.batch_size,
                                         min((i+1)*data_loader.batch_size, len(data_loader.dataset))))
            protein_ids.extend(original_df.iloc[batch_indices]['Accession'].values)
    
    true_families = dataset.label_encoder.inverse_transform(true_labels)
    pred_families = dataset.label_encoder.inverse_transform(predictions)
    
    results_df = pd.DataFrame({
        'Protein': protein_ids,
        'True_Family': true_families,
        'Predicted_Family': pred_families,
        'Confidence': confidences,
        'Index': [data_loader.dataset.indices[i] if hasattr(data_loader.dataset, 'indices') 
                  else i for i in range(len(true_labels))]
    })
    
    # Calculate metrics
    for idx, row in results_df.iterrows():
        true_fam = row['True_Family']
        pred_fam = row['Predicted_Family']
        data_idx = row['Index']
        
        # For standard metrics (original test set)
        if is_negative_control is None or not is_negative_control.get(data_idx, False):
            family_metrics[true_fam]['test_count'] += 1
            
            if true_fam == pred_fam:
                family_metrics[true_fam]['correct'] += 1
            else:
                error_type = analyze_misclassification_type(true_fam, pred_fam)
                if error_type == 'same_superfamily':
                    family_metrics[true_fam]['same_superfamily_errors'] += 1
                else:
                    family_metrics[true_fam]['different_superfamily_errors'] += 1
                    
                family_metrics[true_fam]['misclassified'].append({
                    'Protein': row['Protein'],
                    'Predicted_as': pred_fam,
                    'Confidence': row['Confidence'],
                    'Error_Type': error_type
                })
        
        # For binary classification metrics (with negative controls)
        if is_negative_control is not None and family_test_mapping is not None:
            # Process each family's test set (positive and negative examples)
            for fam, mapping in family_test_mapping.items():
                # Check if this protein is part of this family's test set
                if data_idx in mapping['positive']:
                    # This is a positive example for this family
                    if pred_fam == fam:
                        family_metrics[fam]['TP'] += 1  # Correctly predicted as this family
                    else:
                        family_metrics[fam]['FN'] += 1  # Should be this family but predicted as another
                
                elif data_idx in mapping['negative']:
                    # This is a negative example for this family
                    if pred_fam != fam:
                        family_metrics[fam]['TN'] += 1  # Correctly predicted as not this family
                    else:
                        family_metrics[fam]['FP'] += 1  # Should not be this family but predicted as it
    
    # Create detailed report
    family_report = {}
    for family, metrics in family_metrics.items():
        test_count = metrics['test_count'] # Original test samples for this family
        correct = metrics['correct'] # Correct predictions on original test samples
        accuracy_original_test_set = (correct / test_count * 100) if test_count > 0 else 0.0
        
        # Calculate binary classification metrics if we have negative controls
        precision = 0
        recall = 0
        specificity = 0
        f1_score = 0
        
        if metrics['TP'] + metrics['FP'] > 0:
            precision = metrics['TP'] / (metrics['TP'] + metrics['FP'])
        
        if metrics['TP'] + metrics['FN'] > 0:
            recall = metrics['TP'] / (metrics['TP'] + metrics['FN'])
        
        if metrics['TN'] + metrics['FP'] > 0:
            specificity = metrics['TN'] / (metrics['TN'] + metrics['FP'])
        
        if precision + recall > 0:
            f1_score = 2 * precision * recall / (precision + recall)
        
        # Accuracy including negative controls for this family's binary classification context
        num_instances_binary = metrics['TP'] + metrics['TN'] + metrics['FP'] + metrics['FN']
        accuracy_incl_negative_controls = (metrics['TP'] + metrics['TN']) / num_instances_binary if num_instances_binary > 0 else 0.0
        
        family_report[family] = {
            'Size': metrics['size'],
            'Train_Samples': metrics['train_count'],
            'Test_Samples': test_count,
            'Negative_Controls': metrics['negative_count'],
            'Correct_Predictions': correct,
            'Accuracy_Original_Test_Set': accuracy_original_test_set,
            'Same_Superfamily_Errors': metrics['same_superfamily_errors'],
            'Different_Superfamily_Errors': metrics['different_superfamily_errors'],
            'Misclassified_Details': metrics['misclassified'],
            # Binary classification metrics
            'TP': metrics['TP'],
            'FP': metrics['FP'],
            'TN': metrics['TN'],
            'FN': metrics['FN'],
            'Precision': precision,
            'Recall': recall,
            'Specificity': specificity,
            'F1_Score': f1_score,
            'Accuracy_incl_Negative_Controls': accuracy_incl_negative_controls
        }
    
    return family_report, results_df

# Create dataset
print("\n=== Creating Protein Dataset ===")
print("Processing protein features and encoding labels...")
dataset = ProteinDataset(df)

# Use custom split with negative controls
train_indices, test_indices_with_negatives, is_negative_control, family_test_mapping = custom_split_dataset_with_negatives(df)

# Create custom subsets
train_dataset = torch.utils.data.Subset(dataset, train_indices)
val_dataset = torch.utils.data.Subset(dataset, test_indices_with_negatives)

# Calculate class weights for loss function
all_labels = dataset.labels.numpy()
unique_labels, label_counts = np.unique(all_labels, return_counts=True)
class_weights = 1. / label_counts
class_weights = class_weights / class_weights.sum()  # normalize
class_weights = torch.FloatTensor(class_weights)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Initialize model, loss function, and optimizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_dim = dataset.features.shape[1]
model = ImprovedProteinClassifier(
    input_dim=input_dim,
    num_classes=len(dataset.label_encoder.classes_),
    hidden_dims=[512, 256, 128]
).to(device)

# Use weighted loss for imbalanced classes
class_weights = class_weights.to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)

# Optimizer with weight decay and learning rate scheduler
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=5
)

# Training loop with early stopping and time tracking
num_epochs = 200
best_val_acc = 0
patience = 15
patience_counter = 0
best_model_state = None
best_epoch = 0
history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

print("\n=== Starting Model Training ===")
print("-" * 50)

start_time = time.time()

for epoch in range(num_epochs):
    epoch_start_time = time.time()
    
    # Training phase
    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0
    
    for batch in train_loader:
        features = batch['features'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        
        # L2 regularization
        l2_lambda = 0.001
        l2_reg = torch.tensor(0.).to(device)
        for param in model.parameters():
            l2_reg += torch.norm(param)
        loss += l2_lambda * l2_reg
        
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        train_total += labels.size(0)
        train_correct += predicted.eq(labels).sum().item()
    
    # Validation phase
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for batch in val_loader:
            features = batch['features'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()
    
    train_acc = 100. * train_correct / train_total
    val_acc = 100. * val_correct / val_total
    
    # Update learning rate
    scheduler.step(val_acc)
    
    # Record history
    history['train_loss'].append(train_loss/len(train_loader))
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss/len(val_loader))
    history['val_acc'].append(val_acc)
    
    epoch_end_time = time.time()
    epoch_duration = epoch_end_time - epoch_start_time
    
    print(f'Epoch [{epoch+1}/{num_epochs}] - Time: {epoch_duration:.2f}s')
    print(f'Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%')
    print(f'Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%')
    print('-' * 50)
    
    # Save best model and check early stopping
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        # Store best model in memory instead of saving to file
        best_model_state = model.state_dict().copy()
        best_epoch = epoch
        print(f"New best model found at epoch {epoch+1} with validation accuracy: {val_acc:.2f}%")
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f'Early stopping triggered after {epoch+1} epochs')
            break

total_time = time.time() - start_time
print(f'Total training time: {total_time:.2f}s')

# Plot training history
try:
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.title('Accuracy History')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    # Save the plot to model_results_family folder
    plot_path = os.path.join(results_dir, f'training_history_family.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free memory
    print(f"Training history plot saved as '{plot_path}'")
    # Force terminal output for completion message
    logger.terminal.write(f"Training history plot completed: {plot_path}\n")
    logger.terminal.flush()
except Exception as e:
    print(f"Warning: Could not create or save training history plot: {e}")
    plt.close()  # Make sure to close any open figures

# Load best model for evaluation
print(f"\nUsing best model from epoch {best_epoch+1} with validation accuracy: {best_val_acc:.2f}%")
model.load_state_dict(best_model_state)

# Optional: Try to save the model only at the end
try:
    model_path = os.path.join(results_dir, 'best_protein_classifier_family.pth')
    torch.save({
        'epoch': best_epoch,
        'model_state_dict': best_model_state,
        'best_val_acc': best_val_acc,
    }, model_path)
    print(f"Model saved to {model_path}")
except Exception as e:
    print(f"Warning: Could not save model to file: {e}")
    print("Continuing with in-memory model for evaluation...")

# Detailed evaluation
print("\nPerforming detailed evaluation...")
family_report, results_df = evaluate_model_detailed(model, val_loader, dataset, device, df, train_indices, 
                                                    is_negative_control, family_test_mapping)

# After the evaluation part, modify the printing section:

print("\n=== Detailed Family Classification Report ===")
print("-" * 100)

# Store different superfamily errors for later reporting
different_superfamily_errors_list = []

for family, metrics in family_report.items():
    total_size = metrics['Size']
    print(f"\nFamily: {family}")
    print(f"Total Size: {total_size} members")
    
    # Print data split information based on size
    if total_size == 1:
        print("Data Split: Single member (used in both training and testing)")
    elif total_size == 2:
        print("Data Split: Two members (1 for training, 1 for testing)")
    else:
        print(f"Data Split: {total_size} members (80% training, 20% testing)")
    
    print("\nTraining Set Statistics:")
    print(f"  - Number of training proteins: {metrics['Train_Samples']}")
    
    # Print training proteins details if we have the mapping
    if family_test_mapping:
        # Get all training proteins for this family
        training_proteins = []
        for idx in train_indices:
            if idx < len(df):  # Ensure index is valid
                protein = df.iloc[idx]
                if protein['Family'] == family:
                    training_proteins.append((idx, protein))
        
        if training_proteins:
            print("    Training Protein" + ("s:" if len(training_proteins) > 1 else ":"))
            for idx, protein in training_proteins:
                print(f"      - Accession: {protein['Accession']} | Subfamily: {protein['Subfamily']} | Family: {protein['Family']}")
    
    print("\nTesting Set Statistics:")
    print(f"  - Number of test proteins: {metrics['Test_Samples']}")
    
    # Print test proteins details if we have the mapping
    if family_test_mapping and family in family_test_mapping:
        positive_indices = family_test_mapping[family]['positive']
        if positive_indices:
            print("    Test Protein" + ("s:" if len(positive_indices) > 1 else ":"))
            for idx in positive_indices:
                protein = df.iloc[idx]
                print(f"      - Accession: {protein['Accession']} | Subfamily: {protein['Subfamily']} | Family: {protein['Family']}")
    
    print(f"  - Number of negative controls: {metrics['Negative_Controls']}")
    
    # Print negative controls details if we have the mapping
    if family_test_mapping and family in family_test_mapping:
        negative_indices = family_test_mapping[family]['negative']
        if negative_indices:
            print("    Negative Controls:")
            for i, idx in enumerate(negative_indices, 1):
                protein = df.iloc[idx]
                print(f"      {i}. Accession: {protein['Accession']} | Subfamily: {protein['Subfamily']} | Family: {protein['Family']}")
    
    print(f"  - Correct predictions: {metrics['Correct_Predictions']}")
    print(f"  - Accuracy (Original Test Set): {metrics['Accuracy_Original_Test_Set']:.2f}%")
    
    # Binary classification metrics with negative controls
    print("\nBinary Classification Metrics (with negative controls):")
    print(f"  - True Positives (TP): {metrics['TP']}")
    print(f"  - False Positives (FP): {metrics['FP']}")
    print(f"  - True Negatives (TN): {metrics['TN']}")
    print(f"  - False Negatives (FN): {metrics['FN']}")
    print(f"  - Precision: {metrics['Precision']:.4f}")
    print(f"  - Recall/Sensitivity: {metrics['Recall']:.4f}")
    print(f"  - Specificity: {metrics['Specificity']:.4f}")
    print(f"  - F1 Score: {metrics['F1_Score']:.4f}")
    print(f"  - Accuracy (incl. Negative Controls): {metrics['Accuracy_incl_Negative_Controls']:.4f}")
    
    misclassified_count = len(metrics['Misclassified_Details'])
    if misclassified_count > 0:
        print("\nMisclassification Analysis:")
        print(f"  - Total misclassifications: {misclassified_count}")
        print(f"  - Same superfamily errors: {metrics['Same_Superfamily_Errors']}")
        print(f"  - Different superfamily errors: {metrics['Different_Superfamily_Errors']}")
        
        print("\nMisclassified Proteins Details:")
        for misc in metrics['Misclassified_Details']:
            print(f"  - Protein: {misc['Protein']}")
            print(f"    Predicted as: {misc['Predicted_as']}")
            print(f"    Confidence: {misc['Confidence']:.4f}")
            print(f"    Error Type: {misc['Error_Type']}")
            
            # Collect different superfamily errors
            if misc['Error_Type'] == 'different_superfamily':
                different_superfamily_errors_list.append({
                    'True_Family': family,
                    'Protein': misc['Protein'],
                    'Predicted_as': misc['Predicted_as'],
                    'Confidence': misc['Confidence']
                })
    print("-" * 100)

# Calculate statistics
total_test = sum(m['Test_Samples'] for m in family_report.values())
total_correct = sum(m['Correct_Predictions'] for m in family_report.values())
total_misclassifications = sum(len(m['Misclassified_Details']) for m in family_report.values())
total_same_superfamily_errors = sum(m['Same_Superfamily_Errors'] for m in family_report.values())
total_different_superfamily_errors = sum(m['Different_Superfamily_Errors'] for m in family_report.values())

# Add Confidence Threshold Analysis
print("\n=== Confidence Threshold Analysis ===")
thresholds = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]
confidence_analysis_data = []
# Calculate total number of original test proteins
total_original_test_proteins_count = sum(m['Test_Samples'] for m in family_report.values() if m['Test_Samples'] > 0)


for threshold in thresholds:
    # Filter results based on confidence threshold
    if threshold == 0.0:
        retained_df_all = results_df # Full combined test set
        threshold_label = "0.0 (Full Test Set)"
    else:
        retained_df_all = results_df[results_df['Confidence'] >= threshold]
        threshold_label = f"{threshold:.1f}"

    # Filter for original test proteins from the retained set
    retained_df_original_test = retained_df_all[~retained_df_all['Index'].apply(lambda idx: is_negative_control.get(idx, False))]
    samples_retained_original_test = len(retained_df_original_test)
    
    if total_original_test_proteins_count > 0:
        percent_retained_vs_original_test_set = (samples_retained_original_test / total_original_test_proteins_count) * 100
    else:
        percent_retained_vs_original_test_set = 0.0

    if samples_retained_original_test > 0:
        family_correct_on_original_retained = (retained_df_original_test['True_Family'] == retained_df_original_test['Predicted_Family']).sum()
        family_accuracy_on_original_retained = (family_correct_on_original_retained / samples_retained_original_test) * 100
    else:
        family_accuracy_on_original_retained = 0.0

    confidence_analysis_data.append([
        threshold_label,
        samples_retained_original_test,
        f"{percent_retained_vs_original_test_set:.1f}",
        f"{family_accuracy_on_original_retained:.2f}"
    ])

confidence_table_str = tabulate(confidence_analysis_data,
                           headers=["Confidence Threshold", "Original Test Proteins Above Threshold", "% of Original Test Set Retained", "Family Accuracy (Original Test Set, %)"],
                           tablefmt="grid")
print(confidence_table_str)

# Add column definitions
confidence_column_definitions_str = """
Column Definitions:
* Confidence Threshold: The threshold value applied. '0.0' indicates the evaluation on the complete test set without filtering by confidence.
* Original Test Proteins Above Threshold: The absolute number of *original test proteins* (excluding negative controls) whose model predictions had a confidence score >= the specified threshold.
* % of Original Test Set Retained: The percentage of the *total original test proteins* that were retained (calculated as 'Original Test Proteins Above Threshold' / Total Original Test Proteins * 100).
* Family Accuracy (Original Test Set, %): The accuracy of Family-level classification calculated *only* on the *original test proteins* that were retained above the specified threshold.
"""
print(confidence_column_definitions_str)

# Calculate overall binary classification metrics
total_tp = sum(m['TP'] for m in family_report.values())
total_fp = sum(m['FP'] for m in family_report.values())
total_tn = sum(m['TN'] for m in family_report.values())
total_fn = sum(m['FN'] for m in family_report.values())

overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
overall_specificity = total_tn / (total_tn + total_fp) if (total_tn + total_fp) > 0 else 0
overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
overall_accuracy_binary = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn) if (total_tp + total_tn + total_fp + total_fn) > 0 else 0

# Prepare Overall Classification Statistics string
classification_stats_list = [
    ["Total Test Proteins (Original Set)", total_test],
    ["Total Correct Predictions (Original Set)", total_correct],
    ["Overall Accuracy (Original Test Set)", f"{(total_correct/total_test*100):.2f}%"]
]
overall_classification_stats_str = "\n=== Overall Classification Statistics (Original Test Set) ===\n"
overall_classification_stats_str += tabulate(classification_stats_list, headers=["Metric", "Value"], tablefmt="grid")

# Prepare Binary Classification Metrics string
binary_stats_list = [
    ["True Positives (TP)", total_tp],
    ["False Positives (FP)", total_fp],
    ["True Negatives (TN)", total_tn],
    ["False Negatives (FN)", total_fn],
    ["Precision", f"{overall_precision:.4f}"],
    ["Recall/Sensitivity", f"{overall_recall:.4f}"],
    ["Specificity", f"{overall_specificity:.4f}"],
    ["F1 Score", f"{overall_f1:.4f}"],
    ["Accuracy (incl. Negative Controls)", f"{overall_accuracy_binary:.4f}"]
]
binary_classification_metrics_str = "\n=== Binary Classification Metrics with Negative Controls ===\n"
binary_classification_metrics_str += tabulate(binary_stats_list, headers=["Metric", "Value"], tablefmt="grid")

# Prepare Misclassification Statistics string
misclassification_stats_str = "\n=== Misclassification Statistics (Original Test Set) ===\n"
if total_misclassifications > 0:
    misclassification_stats_list = [
        ["Total Misclassifications", total_misclassifications, "100%"],
        ["Same Superfamily Errors", total_same_superfamily_errors, f"{(total_same_superfamily_errors/total_misclassifications*100):.2f}%"],
        ["Different Superfamily Errors", total_different_superfamily_errors, f"{(total_different_superfamily_errors/total_misclassifications*100):.2f}%"]
    ]
    misclassification_stats_str += tabulate(misclassification_stats_list, headers=["Error Type", "Count", "Percentage"], tablefmt="grid")
else:
    misclassification_stats_str += "No misclassifications found."

# Save confidence analysis and other stats to text file
try:
    classification_stats_path = os.path.join(results_dir, 'classification_stats_family.txt')
    with open(classification_stats_path, 'w', encoding='utf-8') as f:
        f.write("=== Confidence Threshold Analysis ===\n\n")
        f.write(confidence_table_str)
        f.write("\n\n")
        f.write(confidence_column_definitions_str)
        f.write("\n\n")
        f.write(overall_classification_stats_str)
        f.write("\n\n")
        f.write(binary_classification_metrics_str)
        f.write("\n\n")
        f.write(misclassification_stats_str)
    print(f"Confidence threshold analysis and overall stats saved to: {classification_stats_path}")
except Exception as e:
    print(f"Warning: Could not save confidence analysis and overall stats: {e}")

print("\n=== Different Superfamily Error Details ===")
if different_superfamily_errors_list:
    error_details = [[error['True_Family'], 
                     error['Protein'], 
                     error['Predicted_as'], 
                     f"{error['Confidence']:.4f}"] 
                    for error in different_superfamily_errors_list]
    print(tabulate(error_details, 
                  headers=["True Family", "Protein", "Predicted As", "Confidence"], 
                  tablefmt="grid"))
else:
    print("No different superfamily errors found.")

print("-" * 100)

# Save detailed results to CSV in model_results_family folder
try:
    results_csv_path = os.path.join(results_dir, f'detailed_classification_results_family.csv')
    results_df.to_csv(results_csv_path, index=False)
    print(f"\nDetailed results saved to: {results_csv_path}")

    # Save binary classification metrics to CSV
    binary_metrics_df = pd.DataFrame([{
        'Family': family,
        'TP': metrics['TP'],
        'FP': metrics['FP'],
        'TN': metrics['TN'],
        'FN': metrics['FN'],
        'Precision': metrics['Precision'],
        'Recall': metrics['Recall'],
        'Specificity': metrics['Specificity'],
        'F1_Score': metrics['F1_Score'],
        'Test_Samples': metrics['Test_Samples'],
        'Negative_Controls': metrics['Negative_Controls']
    } for family, metrics in family_report.items()])

    binary_metrics_path = os.path.join(results_dir, f'binary_classification_metrics_family.csv')
    binary_metrics_df.to_csv(binary_metrics_path, index=False)
    print(f"Binary classification metrics saved to: {binary_metrics_path}")

    # Force terminal output for completion messages
    logger.terminal.write(f"Classification results CSV completed: {binary_metrics_path}\n")
except Exception as e:
    print(f"Warning: Could not save CSV results: {e}")

# Force terminal output for completion messages
logger.terminal.write(f"Training log completed: {log_file}\n")
logger.terminal.write(f"Classification report completed: {os.path.join(results_dir, 'classification_report_family.txt')}\n")
logger.terminal.write("="*80 + "\n")
logger.terminal.write(f"All results completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
logger.terminal.write(f"All files saved to: {results_dir}\n")
logger.terminal.flush()

# Close the logger
print("="*80)
print(f"Training completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"All results saved to: {results_dir}")
sys.stdout = logger.terminal
logger.close()