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


# Create model_results directory outside scripts folder if it doesn't exist
results_dir = os.path.join('..', 'model_results')
try:
    os.makedirs(results_dir, exist_ok=True)
    print(f"Results directory created/verified at: {results_dir}")
except Exception as e:
    print(f"Warning: Could not create results directory: {e}")
    results_dir = '.'  # Fallback to current directory
    print(f"Using current directory for results instead: {results_dir}")

# Set up logging to capture terminal output
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # Keep timestamp for reference
log_file = os.path.join(results_dir, 'training_log.txt')  # Remove timestamp from filename

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

    def write(self, message):
        # Always write to log file
        self.log.write(message)
        self.log.flush()
        
        # Check if we should enter silent mode
        if "=== Detailed Subfamily Classification Report ===" in message:
            self.silent_mode = True
        
        # Write to terminal only if not in silent mode
        if not self.silent_mode:
            self.terminal.write(message)
            self.terminal.flush()

    def flush(self):
        self.log.flush()
        if not self.silent_mode:
            self.terminal.flush()

    def close(self):
        self.log.close()

# Start logging
logger = Logger(log_file)
sys.stdout = logger

print(f"Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Results will be saved to: {results_dir}")
print("="*80)

# Load data
data_path = os.path.join('..', 'data_source', 'data_new.csv')
df = pd.read_csv(data_path)


class ProteinDataset(Dataset):
    def __init__(self, df, max_domains=50):
        self.max_domains = max_domains
        
        # Convert string representations of lists to actual lists
        df['Domains'] = df['Domains'].apply(ast.literal_eval)
        df['Seperators'] = df['Seperators'].apply(ast.literal_eval)
        
        # Create label encoder for subfamilies
        self.label_encoder = LabelEncoder()
        encoded_labels = self.label_encoder.fit_transform(df['Subfamily'])
        
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
            
            self.features.append(final_features)
            self.labels.append(encoded_labels[_])
        
        # Convert to tensors and normalize features
        self.features = torch.FloatTensor(self.features)
        self.labels = torch.LongTensor(self.labels)
        
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

def analyze_misclassification_type(true_subfamily, pred_subfamily):
    """
    Analyze if misclassification is within same family or different family
    Returns: 'same_family' or 'different_family'
    """
    true_family = '.'.join(true_subfamily.split('.')[:3])
    pred_family = '.'.join(pred_subfamily.split('.')[:3])
    
    if true_family == pred_family:
        return 'same_family'
    return 'different_family'

def custom_split_dataset(df):
    """
    Implements custom splitting strategy based on subfamily size:
    - 1 member: put in both train and test
    - 2 members: split 1:1
    - >2 members: split 80:20
    """
    print("Starting data splitting process...")
    train_indices = []
    test_indices = []
    
    # Group by subfamily to handle each case
    subfamily_counts = {}
    for subfamily, group in df.groupby('Subfamily'):
        indices = group.index.tolist()
        n_samples = len(indices)
        subfamily_counts[subfamily] = n_samples
        
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
    For each subfamily in the test set, generate negative control proteins from other superfamilies.
    
    Returns:
    - negative_control_indices: Dictionary mapping subfamily to list of negative control indices
    - subfamily_to_test_indices: Dictionary mapping subfamily to its test indices
    """
    print("Generating negative control sets...")
    # Get subfamily for each test index
    subfamily_to_test_indices = defaultdict(list)
    for idx in test_indices:
        subfamily = df.iloc[idx]['Subfamily']
        subfamily_to_test_indices[subfamily].append(idx)
    
    # Extract family prefix (first three parts) from each subfamily
    def get_family_prefix(subfamily):
        return '.'.join(subfamily.split('.')[:3])
    
    # Create mapping from subfamily to family
    subfamily_to_family = {subfamily: get_family_prefix(subfamily) for subfamily in df['Subfamily'].unique()}
    
    # Create mapping from family to superfamily
    family_superfamily_map = {}
    for subfamily in df['Subfamily'].unique():
        family = subfamily_to_family[subfamily]
        if family in family_to_superfamily:
            family_superfamily_map[family] = family_to_superfamily[family]
    
    print(f"Found {len(family_superfamily_map)} families with superfamily assignments")
    
    # Generate negative controls for each subfamily
    negative_control_indices = {}
    
    for subfamily, subfamily_test_indices in subfamily_to_test_indices.items():
        family = subfamily_to_family[subfamily]
        target_superfamily = family_superfamily_map.get(family)
        
        # Determine how many negative controls we need
        n_test = len(subfamily_test_indices)
        n_negative = max(n_test, 5)  # At least 5 negative controls
        
        # Find eligible proteins from other superfamilies
        eligible_indices = []
        
        for idx, row in df.iterrows():
            if idx in train_indices:  # Skip training proteins
                continue
                
            other_subfamily = row['Subfamily']
            other_family = subfamily_to_family[other_subfamily]
            
            # Skip if same subfamily
            if other_subfamily == subfamily:
                continue
                
            # If target subfamily has no superfamily, only use proteins from families with superfamily assignments
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
            negative_control_indices[subfamily] = random.sample(eligible_indices, n_negative)
        else:
            # If not enough eligible proteins, use all available
            negative_control_indices[subfamily] = eligible_indices
            print(f"Warning: Not enough negative controls for subfamily {subfamily}. "
                  f"Needed {n_negative}, found {len(eligible_indices)}.")
    
    return negative_control_indices, subfamily_to_test_indices

def custom_split_dataset_with_negatives(df):
    """
    Creates train and test splits with negative controls added to test set.
    
    Returns:
    - train_indices: List of indices for training
    - test_indices_with_negatives: List of indices for testing (includes original test + negative controls)
    - is_negative_control: Dictionary mapping test index to boolean (True if negative control)
    - subfamily_test_mapping: Dictionary mapping subfamily to its test indices (both positive and negative)
    """
    print("\n=== Starting Data Preparation Process ===")
    # Get basic train/test split
    train_indices, test_indices = custom_split_dataset(df)
    
    # Generate negative controls
    negative_control_dict, subfamily_to_test_indices = generate_negative_controls(df, test_indices, train_indices)
    
    # Create combined test set with negative controls
    test_indices_with_negatives = test_indices.copy()
    is_negative_control = {idx: False for idx in test_indices}  # Track which are negative controls
    
    # Create mapping from subfamily to all its test indices (positive and negative)
    subfamily_test_mapping = {}
    
    for subfamily, subfamily_test_indices in subfamily_to_test_indices.items():
        negative_indices = negative_control_dict.get(subfamily, [])
        
        # Add negative controls to test set
        for idx in negative_indices:
            if idx not in test_indices_with_negatives:  # Avoid duplicates
                test_indices_with_negatives.append(idx)
                is_negative_control[idx] = True
        
        # Store mapping of subfamily to all its test indices
        subfamily_test_mapping[subfamily] = {
            'positive': subfamily_test_indices,
            'negative': negative_indices
        }
    
    print("=== Data Preparation Complete ===\n")
    
    return train_indices, test_indices_with_negatives, is_negative_control, subfamily_test_mapping

def evaluate_model_detailed(model, data_loader, dataset, device, original_df, train_indices, is_negative_control=None, subfamily_test_mapping=None):
    model.eval()
    predictions = []
    true_labels = []
    confidences = []
    protein_ids = []
    subfamily_metrics = defaultdict(lambda: {
        'train_count': 0,
        'test_count': 0,
        'correct': 0,
        'size': 0,  # Total size of subfamily
        'misclassified': [],
        'same_family_errors': 0,
        'different_family_errors': 0,
        # New metrics for binary classification
        'TP': 0,  # True Positives
        'FP': 0,  # False Positives
        'TN': 0,  # True Negatives
        'FN': 0,  # False Negatives
        'negative_count': 0  # Number of negative controls
    })
    
    # Calculate total size of each subfamily
    subfamily_counts = original_df['Subfamily'].value_counts()
    for subfamily, count in subfamily_counts.items():
        subfamily_metrics[subfamily]['size'] = count
    
    # Count training samples
    for idx in train_indices:
        subfamily = original_df.iloc[idx]['Subfamily']
        subfamily_metrics[subfamily]['train_count'] += 1
    
    # Count negative controls if provided
    if subfamily_test_mapping:
        for subfamily, mapping in subfamily_test_mapping.items():
            subfamily_metrics[subfamily]['negative_count'] = len(mapping['negative'])
    
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
    
    true_subfamilies = dataset.label_encoder.inverse_transform(true_labels)
    pred_subfamilies = dataset.label_encoder.inverse_transform(predictions)
    
    results_df = pd.DataFrame({
        'Protein': protein_ids,
        'True_Subfamily': true_subfamilies,
        'Predicted_Subfamily': pred_subfamilies,
        'Confidence': confidences,
        'Index': [data_loader.dataset.indices[i] if hasattr(data_loader.dataset, 'indices') 
                  else i for i in range(len(true_labels))]
    })
    
    # Calculate metrics
    for idx, row in results_df.iterrows():
        true_sf = row['True_Subfamily']
        pred_sf = row['Predicted_Subfamily']
        data_idx = row['Index']
        
        # For standard metrics (original test set)
        if is_negative_control is None or not is_negative_control.get(data_idx, False):
            subfamily_metrics[true_sf]['test_count'] += 1
            
            if true_sf == pred_sf:
                subfamily_metrics[true_sf]['correct'] += 1
            else:
                error_type = analyze_misclassification_type(true_sf, pred_sf)
                if error_type == 'same_family':
                    subfamily_metrics[true_sf]['same_family_errors'] += 1
                else:
                    subfamily_metrics[true_sf]['different_family_errors'] += 1
                    
                subfamily_metrics[true_sf]['misclassified'].append({
                    'Protein': row['Protein'],
                    'Predicted_as': pred_sf,
                    'Confidence': row['Confidence'],
                    'Error_Type': error_type
                })
        
        # For binary classification metrics (with negative controls)
        if is_negative_control is not None and subfamily_test_mapping is not None:
            # Process each subfamily's test set (positive and negative examples)
            for sf, mapping in subfamily_test_mapping.items():
                # Check if this protein is part of this subfamily's test set
                if data_idx in mapping['positive']:
                    # This is a positive example for this subfamily
                    if pred_sf == sf:
                        subfamily_metrics[sf]['TP'] += 1  # Correctly predicted as this subfamily
                    else:
                        subfamily_metrics[sf]['FN'] += 1  # Should be this subfamily but predicted as another
                
                elif data_idx in mapping['negative']:
                    # This is a negative example for this subfamily
                    if pred_sf != sf:
                        subfamily_metrics[sf]['TN'] += 1  # Correctly predicted as not this subfamily
                    else:
                        subfamily_metrics[sf]['FP'] += 1  # Should not be this subfamily but predicted as it
    
    # Create detailed report
    subfamily_report = {}
    for subfamily, metrics in subfamily_metrics.items():
        test_count = metrics['test_count']
        correct = metrics['correct']
        accuracy = (correct / test_count * 100) if test_count > 0 else 0
        
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
        
        subfamily_report[subfamily] = {
            'Size': metrics['size'],
            'Train_Samples': metrics['train_count'],
            'Test_Samples': test_count,
            'Negative_Controls': metrics['negative_count'],
            'Correct_Predictions': correct,
            'Accuracy': accuracy,
            'Same_Family_Errors': metrics['same_family_errors'],
            'Different_Family_Errors': metrics['different_family_errors'],
            'Misclassified_Details': metrics['misclassified'],
            # Binary classification metrics
            'TP': metrics['TP'],
            'FP': metrics['FP'],
            'TN': metrics['TN'],
            'FN': metrics['FN'],
            'Precision': precision,
            'Recall': recall,
            'Specificity': specificity,
            'F1_Score': f1_score
        }
    
    return subfamily_report, results_df

# Create dataset
print("\n=== Creating Protein Dataset ===")
print("Processing protein features and encoding labels...")
dataset = ProteinDataset(df)

# Use custom split with negative controls
train_indices, test_indices_with_negatives, is_negative_control, subfamily_test_mapping = custom_split_dataset_with_negatives(df)

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
    # Save the plot to model_results folder
    plot_path = os.path.join(results_dir, f'training_history.png')
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
    model_path = os.path.join(results_dir, 'best_protein_classifier.pth')
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
subfamily_report, results_df = evaluate_model_detailed(model, val_loader, dataset, device, df, train_indices, 
                                                      is_negative_control, subfamily_test_mapping)

# After the evaluation part, modify the printing section:

print("\n=== Detailed Subfamily Classification Report ===")
print("-" * 100)

# Store different family errors for later reporting
different_family_errors_list = []

for subfamily, metrics in subfamily_report.items():
    total_size = metrics['Size']
    print(f"\nSubfamily: {subfamily}")
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
    
    print("\nTesting Set Statistics:")
    print(f"  - Number of test proteins: {metrics['Test_Samples']}")
    
    # Print test proteins details if we have the mapping
    if subfamily_test_mapping and subfamily in subfamily_test_mapping:
        positive_indices = subfamily_test_mapping[subfamily]['positive']
        if positive_indices:
            print("    Test Protein" + ("s:" if len(positive_indices) > 1 else ":"))
            for idx in positive_indices:
                protein = df.iloc[idx]
                print(f"      - Accession: {protein['Accession']} | Subfamily: {protein['Subfamily']}")
    
    print(f"  - Number of negative controls: {metrics['Negative_Controls']}")
    
    # Print negative controls details if we have the mapping
    if subfamily_test_mapping and subfamily in subfamily_test_mapping:
        negative_indices = subfamily_test_mapping[subfamily]['negative']
        if negative_indices:
            print("    Negative Controls:")
            for i, idx in enumerate(negative_indices, 1):
                protein = df.iloc[idx]
                print(f"      {i}. Accession: {protein['Accession']} | Subfamily: {protein['Subfamily']}")
    
    print(f"  - Correct predictions: {metrics['Correct_Predictions']}")
    print(f"  - Accuracy: {metrics['Accuracy']:.2f}%")
    
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
    
    misclassified_count = len(metrics['Misclassified_Details'])
    if misclassified_count > 0:
        print("\nMisclassification Analysis:")
        print(f"  - Total misclassifications: {misclassified_count}")
        print(f"  - Same family errors: {metrics['Same_Family_Errors']}")
        print(f"  - Different family errors: {metrics['Different_Family_Errors']}")
        
        print("\nMisclassified Proteins Details:")
        for misc in metrics['Misclassified_Details']:
            print(f"  - Protein: {misc['Protein']}")
            print(f"    Predicted as: {misc['Predicted_as']}")
            print(f"    Confidence: {misc['Confidence']:.4f}")
            print(f"    Error Type: {misc['Error_Type']}")
            
            # Collect different family errors
            if misc['Error_Type'] == 'different_family':
                different_family_errors_list.append({
                    'True_Subfamily': subfamily,
                    'Protein': misc['Protein'],
                    'Predicted_as': misc['Predicted_as'],
                    'Confidence': misc['Confidence']
                })
    print("-" * 100)

# Calculate statistics
total_test = sum(m['Test_Samples'] for m in subfamily_report.values())
total_correct = sum(m['Correct_Predictions'] for m in subfamily_report.values())
total_misclassifications = sum(len(m['Misclassified_Details']) for m in subfamily_report.values())
total_same_family_errors = sum(m['Same_Family_Errors'] for m in subfamily_report.values())
total_different_family_errors = sum(m['Different_Family_Errors'] for m in subfamily_report.values())

# Calculate overall binary classification metrics
total_tp = sum(m['TP'] for m in subfamily_report.values())
total_fp = sum(m['FP'] for m in subfamily_report.values())
total_tn = sum(m['TN'] for m in subfamily_report.values())
total_fn = sum(m['FN'] for m in subfamily_report.values())

overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
overall_specificity = total_tn / (total_tn + total_fp) if (total_tn + total_fp) > 0 else 0
overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
overall_accuracy = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn) if (total_tp + total_tn + total_fp + total_fn) > 0 else 0

# Initialize counters for family analysis
perfect_accuracy_single_subfamily = defaultdict(int)
perfect_accuracy_multi_subfamily = defaultdict(int)
imperfect_accuracy_single_subfamily = defaultdict(int)
imperfect_accuracy_multi_subfamily = defaultdict(int)
all_families = set()

# Create a mapping of subfamilies to their families
subfamily_to_family = {}
for subfamily in subfamily_report.keys():
    family = '.'.join(subfamily.split('.')[:3])
    subfamily_to_family[subfamily] = family
    all_families.add(family)

# Count subfamilies per family
family_subfamily_count = defaultdict(int)
for subfamily in subfamily_report.keys():
    family = subfamily_to_family[subfamily]
    family_subfamily_count[family] += 1

# Analyze families based on accuracy and subfamily count
for subfamily, metrics in subfamily_report.items():
    family = subfamily_to_family[subfamily]
    accuracy = metrics['Accuracy']
    
    if accuracy == 100.0:
        if family_subfamily_count[family] == 1:
            perfect_accuracy_single_subfamily[family] += 1
        else:
            perfect_accuracy_multi_subfamily[family] += 1
    else:
        if family_subfamily_count[family] == 1:
            imperfect_accuracy_single_subfamily[family] += 1
        else:
            imperfect_accuracy_multi_subfamily[family] = family_subfamily_count[family]

print("\n=== Overall Classification Statistics ===")
classification_stats = [
    ["Total Test Proteins", total_test],
    ["Total Correct Predictions", total_correct],
    ["Overall Accuracy", f"{(total_correct/total_test*100):.2f}%"]
]
print(tabulate(classification_stats, headers=["Metric", "Value"], tablefmt="grid"))

print("\n=== Binary Classification Metrics with Negative Controls ===")
binary_stats = [
    ["True Positives (TP)", total_tp],
    ["False Positives (FP)", total_fp],
    ["True Negatives (TN)", total_tn],
    ["False Negatives (FN)", total_fn],
    ["Precision", f"{overall_precision:.4f}"],
    ["Recall/Sensitivity", f"{overall_recall:.4f}"],
    ["Specificity", f"{overall_specificity:.4f}"],
    ["F1 Score", f"{overall_f1:.4f}"],
    ["Accuracy", f"{overall_accuracy:.4f}"]
]
print(tabulate(binary_stats, headers=["Metric", "Value"], tablefmt="grid"))

print("\n=== Misclassification Statistics ===")
if total_misclassifications > 0:
    misclassification_stats = [
        ["Total Misclassifications", total_misclassifications, "100%"],
        ["Same Family Errors", total_same_family_errors, f"{(total_same_family_errors/total_misclassifications*100):.2f}%"],
        ["Different Family Errors", total_different_family_errors, f"{(total_different_family_errors/total_misclassifications*100):.2f}%"]
    ]
    print(tabulate(misclassification_stats, headers=["Error Type", "Count", "Percentage"], tablefmt="grid"))

print("\n=== Family Analysis Statistics ===")
family_stats = [
    ["Total Number of Families", len(all_families), "100%"],
    ["Families with Single Subfamily (100% Accuracy)", len(perfect_accuracy_single_subfamily), f"{(len(perfect_accuracy_single_subfamily)/len(all_families)*100):.2f}%"],
    ["Families with Single Subfamily (<100% Accuracy)", len(imperfect_accuracy_single_subfamily), f"{(len(imperfect_accuracy_single_subfamily)/len(all_families)*100):.2f}%"],
    ["Families with Multiple Subfamilies (100% Accuracy)", len(perfect_accuracy_multi_subfamily), f"{(len(perfect_accuracy_multi_subfamily)/len(all_families)*100):.2f}%"],
    ["Families with Multiple Subfamilies (<100% Accuracy)", len(imperfect_accuracy_multi_subfamily), f"{(len(imperfect_accuracy_multi_subfamily)/len(all_families)*100):.2f}%"]
]
print(tabulate(family_stats, headers=["Category", "Count", "Percentage"], tablefmt="grid"))

# Verification of total
total_categorized = (len(perfect_accuracy_single_subfamily) + 
                    len(imperfect_accuracy_single_subfamily) + 
                    len(perfect_accuracy_multi_subfamily) + 
                    len(imperfect_accuracy_multi_subfamily))
if total_categorized != len(all_families):
    print("\nWarning: Family categorization sum doesn't match total families!")
    print(f"Total families: {len(all_families)}, Sum of categories: {total_categorized}")

print("\n=== Different Family Error Details ===")
if different_family_errors_list:
    error_details = [[error['True_Subfamily'], 
                     error['Protein'], 
                     error['Predicted_as'], 
                     f"{error['Confidence']:.4f}"] 
                    for error in different_family_errors_list]
    print(tabulate(error_details, 
                  headers=["True Subfamily", "Protein", "Predicted As", "Confidence"], 
                  tablefmt="grid"))
else:
    print("No different family errors found.")

print("-" * 100)

# Save detailed results to CSV in model_results folder
try:
    results_csv_path = os.path.join(results_dir, f'detailed_classification_results.csv')
    results_df.to_csv(results_csv_path, index=False)
    print(f"\nDetailed results saved to: {results_csv_path}")

    # Save binary classification metrics to CSV
    binary_metrics_df = pd.DataFrame([{
        'Subfamily': subfamily,
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
    } for subfamily, metrics in subfamily_report.items()])

    binary_metrics_path = os.path.join(results_dir, f'binary_classification_metrics.csv')
    binary_metrics_df.to_csv(binary_metrics_path, index=False)
    print(f"Binary classification metrics saved to: {binary_metrics_path}")

    # Force terminal output for completion messages
    logger.terminal.write(f"Classification results CSV completed: {results_csv_path}\n")
    logger.terminal.write(f"Binary classification metrics CSV completed: {binary_metrics_path}\n")
except Exception as e:
    print(f"Warning: Could not save CSV results: {e}")

logger.terminal.write(f"Training log completed: {log_file}\n")
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