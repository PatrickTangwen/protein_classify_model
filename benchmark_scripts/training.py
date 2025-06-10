import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time
import os
import joblib
import matplotlib.pyplot as plt

import config

# (Note: The evaluation functions will be moved to evaluation.py later)
# For now, let's assume a placeholder function
def placeholder_evaluation(model, X_val, y_val, model_type):
    print("Placeholder evaluation complete.")
    return {}

def train_sklearn_model(model, X_train, y_train, output_dir):
    """
    Trains a Scikit-learn model and saves it.
    """
    print(f"Training Scikit-learn model: {model.__class__.__name__}...")
    start_time = time.time()
    
    model.fit(X_train, y_train)
    
    end_time = time.time()
    print(f"Training completed in {end_time - start_time:.2f}s.")
    
    joblib.dump(model, os.path.join(output_dir, 'model.joblib'))
    print(f"Model saved to {os.path.join(output_dir, 'model.joblib')}")
    
    return model

def train_pytorch_model(model_class, X_train, y_train, X_val, y_val, output_dir):
    """
    Trains the PyTorch neural network model, logs progress, and saves artifacts.
    """
    print("Training PyTorch model: ImprovedProteinClassifier...")
    log_file = os.path.join(output_dir, 'training_log.txt')
    
    with open(log_file, 'w') as f:
        f.write(f"Training started at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*50 + "\n")

    start_time = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))
    train_loader = DataLoader(train_dataset, batch_size=config.PYTORCH_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.PYTORCH_BATCH_SIZE, shuffle=False)
    
    input_dim = X_train.shape[1]
    num_classes = len(np.unique(np.concatenate((y_train, y_val))))
    model = model_class(input_dim=input_dim, num_classes=num_classes).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.PYTORCH_LR, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0
    patience_counter = 0
    best_model_state = None

    for epoch in range(config.PYTORCH_EPOCHS):
        epoch_start = time.time()
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        scheduler.step(val_acc)
        
        history['train_loss'].append(train_loss / len(train_loader))
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss / len(val_loader))
        history['val_acc'].append(val_acc)
        
        epoch_duration = time.time() - epoch_start
        log_entry = (
            f"Epoch [{epoch+1}/{config.PYTORCH_EPOCHS}] - Time: {epoch_duration:.2f}s\n"
            f"  Train Loss: {history['train_loss'][-1]:.4f}, Train Acc: {train_acc:.2f}%\n"
            f"  Val Loss: {history['val_loss'][-1]:.4f}, Val Acc: {val_acc:.2f}%\n"
        )
        print(log_entry)
        with open(log_file, 'a') as f:
            f.write(log_entry)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.PYTORCH_PATIENCE:
                print(f'Early stopping at epoch {epoch+1}.')
                with open(log_file, 'a') as f:
                    f.write(f'\nEarly stopping at epoch {epoch+1}.\n')
                break
    
    total_time_taken = time.time() - start_time
    print(f"Training completed in {total_time_taken:.2f}s.")
    
    # Save training history plot
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss History')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Accuracy History')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'training_history.png'))
    plt.close()
    
    model.load_state_dict(best_model_state)
    torch.save(model.state_dict(), os.path.join(output_dir, 'model.pth'))
    print(f"Model and history plot saved to {output_dir}")

    return model

def train_and_evaluate_model(model_name, model_info, data, output_dir):
    """
    A generic trainer function that handles both sklearn and PyTorch models.
    """
    X_train, y_train, X_val, y_val = data['X_train'], data['y_train'], data['X_val'], data['y_val']
    
    if model_info['type'] == 'sklearn':
        trained_model = train_sklearn_model(model_info['model'], X_train, y_train, output_dir)
    elif model_info['type'] == 'pytorch':
        trained_model = train_pytorch_model(model_info['model'], X_train, y_train, X_val, y_val, output_dir)
    else:
        raise ValueError(f"Unknown model type: {model_info['type']}")
        
    return trained_model 