import pandas as pd
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import ast
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, label_binarize
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_fscore_support
from torch.utils.data import WeightedRandomSampler
from collections import defaultdict
from scipy.special import softmax
from tabulate import tabulate
import os
import json

# Load data
df = pd.read_csv("output_real.csv")

# Load superfamily information
with open('superfamily.txt', 'r') as f:
    SUPERFAMILIES = json.loads('[' + f.read() + ']')

def get_superfamily(family):
    """Return the superfamily if the family belongs to one, otherwise return None."""
    family_prefix = '.'.join(family.split('.')[:3])  # Get the family part (e.g., '3.A.1')
    return family_prefix if family_prefix in SUPERFAMILIES else None

def are_same_superfamily(family1, family2):
    """Check if two families belong to the same superfamily."""
    superfamily1 = get_superfamily(family1)
    superfamily2 = get_superfamily(family2)
    return superfamily1 is not None and superfamily1 == superfamily2

def calculate_metrics_superfamily_aware(y_true, y_pred, class_label, label_encoder):
    """
    Calculate classification metrics for a specific class, taking superfamily relationships into account.
    """
    y_true_binary = (y_true == label_encoder.transform([class_label])[0]).astype(int)
    y_pred_binary = (y_pred == label_encoder.transform([class_label])[0]).astype(int)
    
    # Get all unique classes
    all_classes = label_encoder.classes_
    
    # Initialize counters
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    
    for i in range(len(y_true)):
        true_class = label_encoder.inverse_transform([y_true[i]])[0]
        pred_class = label_encoder.inverse_transform([y_pred[i]])[0]
        
        if true_class == class_label:
            if pred_class == class_label:
                tp += 1
            else:
                fn += 1
        else:
            # Only count as TN or FP if the true class is not from the same superfamily
            if not are_same_superfamily(true_class, class_label):
                if pred_class == class_label:
                    fp += 1
                else:
                    tn += 1
    
    # Calculate metrics
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'TP': tp,
        'FP': fp,
        'TN': tn,
        'FN': fn,
        'Specificity': specificity,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    }

def calculate_macro_metrics_superfamily_aware(y_true, y_pred, label_encoder):
    """
    Calculate macro-averaged metrics taking superfamily relationships into account.
    """
    classes = label_encoder.classes_
    metrics_per_class = {}
    
    # Calculate metrics for each class
    for class_label in classes:
        metrics = calculate_metrics_superfamily_aware(y_true, y_pred, class_label, label_encoder)
        metrics_per_class[class_label] = metrics
    
    # Calculate macro averages
    macro_metrics = {
        'Macro_Precision': np.mean([m['Precision'] for m in metrics_per_class.values()]),
        'Macro_Recall': np.mean([m['Recall'] for m in metrics_per_class.values()]),
        'Macro_F1': np.mean([m['F1-Score'] for m in metrics_per_class.values()]),
        'Macro_Specificity': np.mean([m['Specificity'] for m in metrics_per_class.values()])
    }
    
    return metrics_per_class, macro_metrics

class ProteinDataset(Dataset):
    def __init__(self, df, max_domains=50):
        self.max_domains = max_domains
        
        # Convert string representations of lists to actual lists
        df['Domains'] = df['Domains'].apply(ast.literal_eval)
        df['Seperator'] = df['Seperator'].apply(ast.literal_eval)
        
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
                domain_acc = domain[0].split(':')[1]  # Get numeric part of CDD:XXXXX
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
                domain_acc = domain[0].split(':')[1]
                domain_idx = self.domain_vocab[domain_acc]
                
                # Update domain presence
                domain_presence[domain_idx] = 1
                
                # Update position features (normalized by protein length)
                domain_positions[domain_idx] = [
                    domain[1] / row['Protein length'],
                    domain[2] / row['Protein length']
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
            for sep in row['Seperator']:
                # Normalize positions by protein length
                start_norm = sep[1] / row['Protein length']
                end_norm = sep[2] / row['Protein length']
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
    train_indices = []
    test_indices = []
    
    # Group by subfamily to handle each case
    for subfamily, group in df.groupby('Subfamily'):
        indices = group.index.tolist()
        n_samples = len(indices)
        
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
    
    return train_indices, test_indices

def evaluate_model_detailed(model, data_loader, dataset, device, original_df, train_indices):
    print("    [Debug] Entering evaluate_model_detailed...") # DEBUG
    model.eval()
    
    all_true_labels_encoded = []
    all_predictions_encoded = []
    all_probabilities = []
    
    protein_ids_eval = [] # Renamed to avoid conflict
    confidences_eval = [] # Renamed to avoid conflict

    subfamily_metrics = defaultdict(lambda: {
        'train_count': 0,
        'test_count': 0,
        'correct': 0,
        'size': 0,  # Total size of subfamily
        'misclassified': [],
        'same_family_errors': 0,
        'different_family_errors': 0
    })
    
    subfamily_counts = original_df['Subfamily'].value_counts()
    for subfamily, count in subfamily_counts.items():
        subfamily_metrics[subfamily]['size'] = count
    
    for idx in train_indices:
        subfamily = original_df.iloc[idx]['Subfamily']
        subfamily_metrics[subfamily]['train_count'] += 1
    
    print("    [Debug] Starting evaluation loop over data_loader...") # DEBUG
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            if i % 10 == 0: # Print every 10 batches to avoid too much output
                print(f"      [Debug] Processing batch {i+1}/{len(data_loader)}...") # DEBUG
            features = batch['features'].to(device)
            labels = batch['label'].to(device) # These are already encoded
            
            outputs = model(features)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, dim=1)
            
            all_true_labels_encoded.extend(labels.cpu().numpy())
            all_predictions_encoded.extend(predicted.cpu().numpy())
            all_probabilities.append(probs.cpu().numpy())
            
            confidences_eval.extend(confidence.cpu().numpy())
            
            # Get protein IDs for results_df
            if hasattr(data_loader.dataset, 'indices'): # Subset
                current_batch_indices = data_loader.dataset.indices[i*data_loader.batch_size : (i+1)*data_loader.batch_size]
            else: # Full dataset (should not happen with custom split but good practice)
                current_batch_indices = list(range(i*data_loader.batch_size, (i+1)*data_loader.batch_size))
            protein_ids_eval.extend(original_df.iloc[current_batch_indices]['Protein'].values)
    
    print("    [Debug] Evaluation loop finished. Processing results...") # DEBUG
    # Concatenate probabilities from all batches
    all_probabilities_np = np.concatenate(all_probabilities, axis=0)
    
    true_subfamilies_decoded = dataset.label_encoder.inverse_transform(all_true_labels_encoded)
    pred_subfamilies_decoded = dataset.label_encoder.inverse_transform(all_predictions_encoded)
    
    results_df = pd.DataFrame({
        'Protein': protein_ids_eval,
        'True_Subfamily': true_subfamilies_decoded,
        'Predicted_Subfamily': pred_subfamilies_decoded,
        'Confidence': confidences_eval # This is the confidence of the predicted class
    })
    
    for i in range(len(true_subfamilies_decoded)):
        true_sf = true_subfamilies_decoded[i]
        pred_sf = pred_subfamilies_decoded[i]
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
                'Protein': results_df.iloc[i]['Protein'],
                'Predicted_as': pred_sf,
                'Confidence': results_df.iloc[i]['Confidence'],
                'Error_Type': error_type
            })
    
    subfamily_report_summary = {}
    for subfamily, metrics in subfamily_metrics.items():
        test_count = metrics['test_count']
        correct = metrics['correct']
        accuracy = (correct / test_count * 100) if test_count > 0 else 0
        
        subfamily_report_summary[subfamily] = {
            'Size': metrics['size'],
            'Train_Samples': metrics['train_count'],
            'Test_Samples': test_count,
            'Correct_Predictions': correct,
            'Accuracy': accuracy,
            'Same_Family_Errors': metrics['same_family_errors'],
            'Different_Family_Errors': metrics['different_family_errors'],
            'Misclassified_Details': metrics['misclassified']
        }
    
    print("    [Debug] Exiting evaluate_model_detailed.") # DEBUG
    return (subfamily_report_summary, results_df, 
            np.array(all_true_labels_encoded), 
            np.array(all_predictions_encoded), 
            all_probabilities_np)

# --- New Helper Functions ---
def plot_confusion_matrix_heatmap(cm, class_names, output_path, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title = title + ' (Normalized)'
    else:
        fmt = 'd'
        title = title
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap=cmap, xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    return output_path

def plot_multiclass_roc_auc(y_true_encoded, y_prob, label_encoder, output_path, title='ROC Curve'):
    """
    Plot only the micro and macro-average ROC curves for overall performance visualization.
    """
    class_labels = label_encoder.classes_
    n_classes = len(class_labels)
    y_true_binarized = label_binarize(y_true_encoded, classes=range(n_classes))
    
    plt.figure(figsize=(10, 8))
    
    # Calculate individual curves (needed for macro-average)
    fpr = dict()
    tpr = dict()
    roc_auc_scores = dict()
    
    # Calculate individual ROC curves (but don't plot them)
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_binarized[:, i], y_prob[:, i])
        roc_auc_scores[class_labels[i]] = auc(fpr[i], tpr[i])
    
    # Micro-average ROC curve
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_binarized.ravel(), y_prob.ravel())
    roc_auc_scores["micro"] = auc(fpr["micro"], tpr["micro"])
    plt.plot(fpr["micro"], tpr["micro"],
             label=f'Micro-average ROC curve (AUC = {roc_auc_scores["micro"]:.2f})',
             color='deeppink', linestyle='-', linewidth=2)
    
    # Macro-average ROC curve
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc_scores["macro"] = auc(fpr["macro"], tpr["macro"])
    plt.plot(fpr["macro"], tpr["macro"],
             label=f'Macro-average ROC curve (AUC = {roc_auc_scores["macro"]:.2f})',
             color='navy', linestyle='-', linewidth=2)
    
    # Plot diagonal line
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.8)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    return output_path, roc_auc_scores

# --- New Helper Function for Macro Specificity ---
def calculate_macro_specificity(cm):
    num_classes = cm.shape[0]
    specificities = []
    for i in range(num_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - (tp + fp + fn)
        
        specificity_class = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        specificities.append(specificity_class)
    
    macro_specificity = np.mean(specificities) if specificities else 0.0
    return macro_specificity
# --- End New Helper Function ---

# Create dataset
dataset = ProteinDataset(df)

# Use custom split
train_indices, test_indices = custom_split_dataset(df)

# Create custom subsets
train_dataset = torch.utils.data.Subset(dataset, train_indices)
val_dataset = torch.utils.data.Subset(dataset, test_indices)

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
    optimizer, mode='max', factor=0.5, patience=5, verbose=True
)

# Training loop with early stopping and time tracking
num_epochs = 200
best_val_acc = 0
patience = 15
patience_counter = 0
history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

start_time = time.time()

# --- Define output directory ---
output_dir = 'old_data'
os.makedirs(output_dir, exist_ok=True)

# --- Define output file names ---
best_model_path = os.path.join(output_dir, 'best_protein_classifier_old_data.pth')
report_filename = os.path.join(output_dir, 'evaluation_report_old_data.txt')
results_csv_filename = os.path.join(output_dir, 'classification_results_old_data.csv')
plot_filename = os.path.join(output_dir, 'training_history_old_data.png')
comprehensive_metrics_filename = os.path.join(output_dir, 'comprehensive_metrics_old_data.txt')
# New filename for overall metrics
overall_metrics_filename = os.path.join(output_dir, 'overall_evaluation_metrics_old_data.txt')
roc_plot_path_in_report = os.path.join(output_dir, 'roc_auc_old_data.png')

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
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_acc': best_val_acc,
        }, best_model_path)
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f'Early stopping triggered after {epoch+1} epochs')
            break

total_time = time.time() - start_time
print(f'Total training time: {total_time:.2f}s')

# Plot training history
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
plt.savefig(plot_filename)
plt.close()

# Load best model for evaluation
checkpoint = torch.load(best_model_path)
model.load_state_dict(checkpoint['model_state_dict'])

# Detailed evaluation
print("\nPerforming detailed evaluation...")
subfamily_report, results_df, y_true_eval, y_pred_eval, y_prob_eval = evaluate_model_detailed(model, val_loader, dataset, device, df, train_indices)

print("    [Debug] Returned from evaluate_model_detailed.") # DEBUG

# --- Comprehensive Evaluation Metrics ---
class_names_eval = dataset.label_encoder.classes_
print("    [Debug] Generated class_names_eval.") # DEBUG

# 1. Classification Report (already includes precision, recall, F1 per class)
print("    [Debug] Calculating classification_report...") # DEBUG
report_str = classification_report(y_true_eval, y_pred_eval, target_names=class_names_eval, zero_division=0)
report_dict = classification_report(y_true_eval, y_pred_eval, target_names=class_names_eval, zero_division=0, output_dict=True) # For overall metrics
print("    [Debug] Calculated classification_report.") # DEBUG

# 2. Confusion Matrix (still needed for macro_specificity)
print("    [Debug] Calculating confusion_matrix...") # DEBUG
cm_eval = confusion_matrix(y_true_eval, y_pred_eval)
print("    [Debug] Calculated confusion_matrix.") # DEBUG

# 3. ROC Curve and AUC (y_prob_eval and roc_auc_scores_eval are still needed for overall AUC)
print("    [Debug] Plotting ROC curve and AUC...") # DEBUG
roc_plot_path, roc_auc_scores_eval = plot_multiclass_roc_auc(y_true_eval, y_prob_eval, dataset.label_encoder,
                                                          os.path.join(output_dir, 'roc_auc_old_data.png'),
                                                          title='ROC Curve - Old Data')
print("    [Debug] Plotted ROC curve and AUC.") # DEBUG

# --- Calculate Overall Metrics for the new file ---
print("    [Debug] Calculating overall metrics...") #DEBUG
micro_precision = report_dict['weighted avg']['precision']
micro_recall = report_dict['weighted avg']['recall']
micro_f1 = report_dict['weighted avg']['f1-score']
macro_precision = report_dict['macro avg']['precision']
macro_recall = report_dict['macro avg']['recall']
macro_f1 = report_dict['macro avg']['f1-score']
macro_specificity = calculate_macro_specificity(cm_eval)
micro_auc = roc_auc_scores_eval.get('micro', 0.0)
macro_auc = roc_auc_scores_eval.get('macro', 0.0)
print("    [Debug] Finished calculating overall metrics.") #DEBUG

# --- Write Overall Metrics to Separate File ---
print(f"    [Debug] Writing overall metrics to {overall_metrics_filename}...") #DEBUG
with open(overall_metrics_filename, 'w') as om_file:
    om_file.write("=== Overall Evaluation Metrics ===\n\n")
    om_file.write(f"Weighted-Averaged Precision: {micro_precision:.4f}\n")
    om_file.write(f"Weighted-Averaged Recall (Sensitivity): {micro_recall:.4f}\n")
    om_file.write(f"Weighted-Averaged F1-Score: {micro_f1:.4f}\n\n")
    om_file.write(f"Macro-Averaged Precision: {macro_precision:.4f}\n")
    om_file.write(f"Macro-Averaged Recall (Sensitivity): {macro_recall:.4f}\n")
    om_file.write(f"Macro-Averaged F1-Score: {macro_f1:.4f}\n")
    om_file.write(f"Macro-Averaged Specificity: {macro_specificity:.4f}\n\n")
    om_file.write(f"Micro-Averaged AUC: {micro_auc:.4f}\n")
    om_file.write(f"Macro-Averaged AUC: {macro_auc:.4f}\n")
print(f"    [Debug] Finished writing overall metrics.") #DEBUG

# --- Write Comprehensive Metrics (Sklearn Report Only) to Separate File ---
print(f"    [Debug] Writing comprehensive sklearn report to {comprehensive_metrics_filename}...") #DEBUG
with open(comprehensive_metrics_filename, 'w') as cm_file:
    cm_file.write("=== Classification Report (Sklearn) ===\n\n")
    cm_file.write(report_str) # report_str is the full classification report string
    cm_file.write("\n")
print(f"    [Debug] Finished writing comprehensive sklearn report.") #DEBUG

# --- Calculate Overall Statistics from Subfamily Report ---
print("    [Debug] Calculating overall statistics from subfamily_report...") # DEBUG
total_test = sum(m['Test_Samples'] for m in subfamily_report.values())
total_correct = sum(m['Correct_Predictions'] for m in subfamily_report.values())
total_misclassifications_sr = sum(len(m['Misclassified_Details']) for m in subfamily_report.values())
total_same_family_errors = sum(m['Same_Family_Errors'] for m in subfamily_report.values())
total_different_family_errors = sum(m['Different_Family_Errors'] for m in subfamily_report.values())
print("    [Debug] Finished calculating overall statistics from subfamily_report.") # DEBUG
# --- End Calculate Overall Statistics ---

# Calculate total test set size
print("    [Debug] Calculating total_test_set_size...") # DEBUG
total_test_set_size = len(val_dataset)

# Add Family columns for analysis
results_df['True_Family'] = results_df['True_Subfamily'].apply(lambda x: '.'.join(x.split('.')[:3]))
results_df['Predicted_Family'] = results_df['Predicted_Subfamily'].apply(lambda x: '.'.join(x.split('.')[:3]))

# Confidence Threshold Analysis
thresholds = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]
confidence_analysis_data = []

for threshold in thresholds:
    # Filter results based on confidence threshold
    if threshold == 0.0:
        retained_df = results_df # Full test set
        threshold_label = "0.0 (Full Test Set)"
    else:
        retained_df = results_df[results_df['Confidence'] >= threshold]
        threshold_label = f"{threshold:.1f}"

    samples_retained = len(retained_df)
    
    if total_test_set_size > 0:
        percent_retained = (samples_retained / total_test_set_size) * 100
    else:
        percent_retained = 0.0

    if samples_retained > 0:
        family_correct = (retained_df['True_Family'] == retained_df['Predicted_Family']).sum()
        subfamily_correct = (retained_df['True_Subfamily'] == retained_df['Predicted_Subfamily']).sum()
        family_accuracy = (family_correct / samples_retained) * 100
        subfamily_accuracy = (subfamily_correct / samples_retained) * 100
    else:
        family_accuracy = 0.0
        subfamily_accuracy = 0.0

    confidence_analysis_data.append([
        threshold_label,
        samples_retained,
        f"{percent_retained:.1f}",
        f"{family_accuracy:.2f}",
        f"{subfamily_accuracy:.2f}"
    ])

confidence_table = tabulate(confidence_analysis_data,
                           headers=["Confidence Threshold", "Samples Retained", "% of Total Test Set", "Family Accuracy (%)", "Subfamily Accuracy (%)"],
                           tablefmt="grid")

# --- Define Column Definitions Text ---
confidence_column_definitions = """
Column Definitions:
* Confidence Threshold: The threshold value applied. '0.0' indicates the evaluation on the complete test set without filtering.
* Samples Retained: The absolute number of test samples whose predictions had a confidence score >= the specified threshold.
* % of Total Test Set: The percentage of the *original total number of test samples* that were retained (calculated as Samples Retained / Total Test Set Size * 100).
* Family Accuracy (%): The accuracy of Family-level classification calculated *only* on the samples retained at this threshold.
* Subfamily Accuracy (%): The accuracy of Subfamily-level classification calculated *only* on the samples retained at this threshold.
"""

# Write all evaluation results to a file
output_filename = report_filename
with open(output_filename, 'w') as f:
    f.write("=== Detailed Subfamily Classification Report ===\n\n")
    f.write("---" * 30 + "\n")

    different_family_errors_list = []

    for subfamily, metrics in subfamily_report.items():
        total_size = metrics['Size']
        f.write(f"\n--- Subfamily: {subfamily} ---\n")
        f.write(f"Total Size: {total_size} members\n\n")

        f.write("Data Split: ")
        if total_size == 1:
            f.write("Single member (used in both training and testing)\n")
        elif total_size == 2:
            f.write("Two members (1 for training, 1 for testing)\n")
        else:
            f.write(f"{total_size} members (80% training, 20% testing)\n")

        f.write("\nTraining Set Statistics:\n")
        f.write(f"- Number of training proteins: {metrics['Train_Samples']}\n")

        f.write("\nTesting Set Statistics:\n")
        f.write(f"- Number of test proteins: {metrics['Test_Samples']}\n")
        f.write(f"- Correct predictions: {metrics['Correct_Predictions']}\n")
        f.write(f"- Accuracy: {metrics['Accuracy']:.2f}%\n")

        misclassified_count = len(metrics['Misclassified_Details'])
        if misclassified_count > 0:
            f.write("\nMisclassification Analysis:\n")
            f.write(f"- Total misclassifications: {misclassified_count}\n")
            f.write(f"- Same family errors: {metrics['Same_Family_Errors']}\n")
            f.write(f"- Different family errors: {metrics['Different_Family_Errors']}\n")

            f.write("\nMisclassified Proteins Details:\n")
            for misc in metrics['Misclassified_Details']:
                f.write(f"- Protein: {misc['Protein']}\n")
                f.write(f"  - Predicted as: {misc['Predicted_as']}\n")
                f.write(f"  - Confidence: {misc['Confidence']:.4f}\n")
                f.write(f"  - Error Type: {misc['Error_Type']}\n")

                if misc['Error_Type'] == 'different_family':
                    different_family_errors_list.append({
                        'True_Subfamily': subfamily,
                        'Protein': misc['Protein'],
                        'Predicted_as': misc['Predicted_as'],
                        'Confidence': misc['Confidence']
                    })
        f.write("\n" + "---" * 30 + "\n")

    f.write("\n=== Overall Classification Statistics (from Subfamily Report) ===\n\n")
    classification_stats = [
        ["Total Test Proteins", total_test],
        ["Total Correct Predictions", total_correct],
        ["Overall Accuracy", f"{(total_correct/total_test*100):.2f}%" if total_test > 0 else "N/A"]
    ]
    f.write(tabulate(classification_stats, headers=["Metric", "Value"], tablefmt="grid"))
    f.write("\n")

    f.write("\n=== Misclassification Statistics (from Subfamily Report) ===\n\n")
    if total_misclassifications_sr > 0:
        misclassification_stats_sr = [
            ["Total Misclassifications", total_misclassifications_sr, "100%"],
            ["Same Family Errors", total_same_family_errors, f"{(total_same_family_errors/total_misclassifications_sr*100):.2f}%"],
            ["Different Family Errors", total_different_family_errors, f"{(total_different_family_errors/total_misclassifications_sr*100):.2f}%"]
        ]
        f.write(tabulate(misclassification_stats_sr, headers=["Error Type", "Count", "Percentage"], tablefmt="grid"))
        f.write("\n")
    else:
        f.write("No misclassifications found (from subfamily report).\n")
    
    f.write("\n=== Family Analysis Statistics (from Subfamily Report) ===\n\n")
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

    family_stats_data = [
        ["Total Number of Families", len(all_families), "100%"],
        ["Families with Single Subfamily (100% Accuracy)", len(perfect_accuracy_single_subfamily), f"{(len(perfect_accuracy_single_subfamily)/len(all_families)*100):.2f}%" if len(all_families)>0 else "N/A"],
        ["Families with Single Subfamily (<100% Accuracy)", len(imperfect_accuracy_single_subfamily), f"{(len(imperfect_accuracy_single_subfamily)/len(all_families)*100):.2f}%" if len(all_families)>0 else "N/A"],
        ["Families with Multiple Subfamilies (100% Accuracy)", len(perfect_accuracy_multi_subfamily), f"{(len(perfect_accuracy_multi_subfamily)/len(all_families)*100):.2f}%" if len(all_families)>0 else "N/A"],
        ["Families with Multiple Subfamilies (<100% Accuracy)", len(imperfect_accuracy_multi_subfamily), f"{(len(imperfect_accuracy_multi_subfamily)/len(all_families)*100):.2f}%" if len(all_families)>0 else "N/A"]
    ]
    f.write(tabulate(family_stats_data, headers=["Category", "Count", "Percentage"], tablefmt="grid"))
    f.write("\n")

    total_categorized = (len(perfect_accuracy_single_subfamily) + len(imperfect_accuracy_single_subfamily) +
                         len(perfect_accuracy_multi_subfamily) + len(imperfect_accuracy_multi_subfamily))
    if total_categorized != len(all_families) and len(all_families) > 0:
        f.write(f"\nWarning: Family categorization sum ({total_categorized}) might not match total families ({len(all_families)}) due to multi-subfamily imperfect accuracy counting.\n")

    f.write("\n=== Different Family Error Details (from Subfamily Report) ===\n\n")
    if different_family_errors_list:
        error_details_data = [[err['True_Subfamily'], err['Protein'], err['Predicted_as'], f"{err['Confidence']:.4f}"] for err in different_family_errors_list]
        f.write(tabulate(error_details_data, headers=["True Subfamily", "Protein", "Predicted As", "Confidence"], tablefmt="grid"))
        f.write("\n")
    else:
        f.write("No different family errors found (from subfamily report).\n")
    f.write("\n" + "---" * 30 + "\n")

    f.write("\n=== Comprehensive Classification Metrics ===\n\n")
    f.write(f"Full Sklearn Classification Report can be found in the file: {os.path.basename(comprehensive_metrics_filename)}\n\n")
    f.write(f"Overall Averaged Metrics (Precision, Recall, F1, Specificity, AUC) can be found in the file: {os.path.basename(overall_metrics_filename)}\n\n")
    
    f.write(f"ROC Curve Plot: {os.path.basename(roc_plot_path_in_report)}\n\n")
    
    f.write("\n=== Confidence Threshold Accuracy Analysis ===\n\n")
    f.write(confidence_table)
    f.write("\n\n" + confidence_column_definitions) 
    f.write("\n")
    f.write("-" * 100 + "\n")

print(f"Evaluation report saved to {output_filename}")

# Save detailed results to CSV
results_df.to_csv(results_csv_filename, index=False)