import numpy as np
import os
from pathlib import Path
import json
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print(f"✅ GPU Available: {torch.cuda.get_device_name(0)}")
    print(f"   CUDA Version: {torch.version.cuda}")
    # Clear any cached memory
    torch.cuda.empty_cache()
    # Get GPU memory info
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"   Total GPU Memory: {total_memory:.2f} GB")
else:
    print("⚠️  No GPU detected. Training will use CPU (slower)")


class ASLDataset(Dataset):
    """PyTorch Dataset with augmentation"""
    
    def __init__(self, sequences, labels, num_classes, augment=False):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.LongTensor(labels)
        self.num_classes = num_classes
        self.augment = augment
    
    def __len__(self):
        return len(self.sequences)
    
    def augment_sequence(self, x):
        """Apply random augmentations"""
        # Small random noise (don't overdo it)
        if np.random.rand() < 0.3:
            noise = torch.randn_like(x) * 0.008
            x = x + noise
        
        # Small random scaling
        if np.random.rand() < 0.3:
            scale = 0.95 + np.random.rand() * 0.1  # 0.95 to 1.05
            x = x * scale
        
        # Time shift
        if np.random.rand() < 0.25:
            shift = np.random.randint(-2, 3)
            x = torch.roll(x, shifts=shift, dims=0)
        
        return x
    
    def __getitem__(self, idx):
        x = self.sequences[idx].clone()
        y = self.labels[idx]
        
        if self.augment:
            x = self.augment_sequence(x)
        
        return x, y


class TemporalConvBlock(nn.Module):
    """Temporal Convolution Block - NO dropout here (feature extraction)"""
    
    def __init__(self, input_channels, output_channels, kernel_size=3):
        super(TemporalConvBlock, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, output_channels, kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(output_channels)
        self.conv2 = nn.Conv1d(output_channels, output_channels, kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(output_channels)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # x: (batch, seq_len, features) -> (batch, features, seq_len)
        x = x.transpose(1, 2)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        # Back to (batch, seq_len, features)
        x = x.transpose(1, 2)
        return x


class BalancedASLModel(nn.Module):
    """
    Enhanced model: TCN + BiLSTM + Attention with strategic dropout
    Architecture: Multi-Stream TCN -> Fusion -> BiLSTM -> Attention -> Classifier
    MEMORY OPTIMIZED VERSION
    """
    
    def __init__(self, input_size, num_classes, hidden_size=192, dropout=0.25):  # Reduced from 256 to 192
        super(BalancedASLModel, self).__init__()
        
        self.pose_size = 132
        self.face_size = 1404
        self.hand_size = 63
        
        # Temporal Convolutional Networks for each modality - NO dropout
        self.pose_tcn = TemporalConvBlock(self.pose_size, hidden_size // 4, kernel_size=3)
        self.face_tcn = TemporalConvBlock(self.face_size, hidden_size // 2, kernel_size=3)
        self.lhand_tcn = TemporalConvBlock(self.hand_size, hidden_size // 8, kernel_size=3)
        self.rhand_tcn = TemporalConvBlock(self.hand_size, hidden_size // 8, kernel_size=3)
        
        # Fusion with light dropout AFTER concatenation
        combined_size = hidden_size // 4 + hidden_size // 2 + hidden_size // 8 + hidden_size // 8
        self.fusion = nn.Linear(combined_size, hidden_size)
        self.fusion_ln = nn.LayerNorm(hidden_size)
        self.fusion_dropout = nn.Dropout(dropout)  # Strategic dropout #1
        
        # Bidirectional LSTM - Reduced to 1 layer to save memory
        self.lstm = nn.LSTM(
            hidden_size, hidden_size, 
            num_layers=1,  # Changed from 2 to 1
            batch_first=True,
            bidirectional=True,
            dropout=0  # No dropout with single layer
        )
        self.lstm_ln = nn.LayerNorm(hidden_size * 2)
        self.lstm_dropout = nn.Dropout(dropout)  # Manual dropout after LSTM
        
        # Attention (simple and effective)
        self.attention_weights = nn.Linear(hidden_size * 2, 1)
        
        # Classifier with strategic dropout
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.dropout1 = nn.Dropout(dropout)  # Strategic dropout #2
        
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.ln2 = nn.LayerNorm(hidden_size // 2)
        self.dropout2 = nn.Dropout(dropout * 0.5)  # Strategic dropout #3 (lighter)
        
        self.fc3 = nn.Linear(hidden_size // 2, num_classes)
        
        # Weight initialization
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    nn.init.kaiming_normal_(param.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    nn.init.constant_(param.data, 0)
    
    def forward(self, x):
        # Split into modalities
        pose = x[:, :, :self.pose_size]
        face = x[:, :, self.pose_size:self.pose_size+self.face_size]
        lhand = x[:, :, self.pose_size+self.face_size:self.pose_size+self.face_size+self.hand_size]
        rhand = x[:, :, self.pose_size+self.face_size+self.hand_size:]
        
        # Apply TCN to each modality (local temporal patterns)
        pose_feat = self.pose_tcn(pose)      # (batch, 30, hidden//4)
        face_feat = self.face_tcn(face)      # (batch, 30, hidden//2)
        lhand_feat = self.lhand_tcn(lhand)   # (batch, 30, hidden//8)
        rhand_feat = self.rhand_tcn(rhand)   # (batch, 30, hidden//8)
        
        # Concatenate and fuse
        combined = torch.cat([pose_feat, face_feat, lhand_feat, rhand_feat], dim=2)
        x = F.relu(self.fusion_ln(self.fusion(combined)))
        x = self.fusion_dropout(x)  # Dropout after fusion
        
        # BiLSTM
        lstm_out, _ = self.lstm(x)
        lstm_out = self.lstm_ln(lstm_out)
        lstm_out = self.lstm_dropout(lstm_out)  # Add dropout after LSTM
        
        # Attention mechanism
        attn_scores = self.attention_weights(lstm_out)
        attn_weights = F.softmax(attn_scores, dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        
        # Classification layers
        x = F.relu(self.ln1(self.fc1(context)))
        x = self.dropout1(x)
        
        x = F.relu(self.ln2(self.fc2(x)))
        x = self.dropout2(x)
        
        x = self.fc3(x)
        
        return x


class ASLModelTrainer:
    """ASL Model Trainer with detailed metrics"""
    
    def __init__(self, data_dir='augmented_data', model_name='asl_balanced_model'):
        self.data_dir = Path(data_dir)
        self.model_name = model_name
        self.model = None
        self.classes = []
        self.label_map = {}
        self.device = device
        
    def load_data(self):
        """Load augmented keypoint data"""
        print("=" * 70)
        print("LOADING ASL TRAINING DATA")
        print("=" * 70)
        
        train_dir = self.data_dir / 'train'
        val_dir = self.data_dir / 'val'
        
        if not train_dir.exists():
            print(f"❌ ERROR: Training directory not found: {train_dir}")
            return None, None, None, None
        
        self.classes = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
        self.label_map = {label: i for i, label in enumerate(self.classes)}
        
        print(f"\n📊 Dataset Information:")
        print(f"  Classes: {len(self.classes)}")
        print(f"  First 10: {self.classes[:10]}")
        
        print("\n📥 Loading training data...")
        X_train, y_train = self._load_from_directory(train_dir)
        
        print("\n📥 Loading validation data...")
        X_val, y_val = self._load_from_directory(val_dir)
        
        # Print class distribution
        print("\n" + "=" * 70)
        print("CLASS DISTRIBUTION ANALYSIS")
        print("=" * 70)
        
        from collections import Counter
        train_dist = Counter(y_train)
        val_dist = Counter(y_val)
        
        print(f"\nTrain samples per class: min={min(train_dist.values())}, max={max(train_dist.values())}, avg={np.mean(list(train_dist.values())):.1f}")
        print(f"Val samples per class: min={min(val_dist.values())}, max={max(val_dist.values())}, avg={np.mean(list(val_dist.values())):.1f}")
        
        # Show classes with least samples
        print(f"\nClasses with LEAST training samples:")
        least_samples = sorted(train_dist.items(), key=lambda x: x[1])[:5]
        for label_idx, count in least_samples:
            print(f"  {self.classes[label_idx]:20s}: {count:4d} samples")
        
        print("\n" + "=" * 70)
        print("DATA LOADING COMPLETE!")
        print("=" * 70)
        print(f"✅ Training samples: {len(X_train)}")
        print(f"✅ Validation samples: {len(X_val)}")
        print(f"✅ Sequence length: {X_train.shape[1]} frames")
        print(f"✅ Features: {X_train.shape[2]}")
        print("=" * 70)
        
        return X_train, y_train, X_val, y_val
    
    def _load_from_directory(self, directory):
        """Load all .npy files from directory"""
        sequences = []
        labels = []
        
        class_dirs = sorted([d for d in directory.iterdir() if d.is_dir()])
        
        for class_dir in class_dirs:
            class_name = class_dir.name
            if class_name not in self.label_map:
                continue
            
            class_label = self.label_map[class_name]
            npy_files = list(class_dir.glob('*.npy'))
            
            print(f"  {class_name:20s}: {len(npy_files):4d} samples", end='\r')
            
            for npy_file in npy_files:
                try:
                    sequence = np.load(npy_file)
                    if sequence.shape[0] != 30 or sequence.shape[1] != 1662:
                        continue
                    if np.isnan(sequence).any() or np.isinf(sequence).any():
                        continue
                    sequences.append(sequence)
                    labels.append(class_label)
                except:
                    continue
        
        print()
        return np.array(sequences), np.array(labels)
    
    def build_model(self, input_size, num_classes):
        """Build balanced model"""
        print("\n" + "=" * 70)
        print("BUILDING BALANCED ASL MODEL")
        print("=" * 70)
        print("Architecture: Multi-Stream TCN + BiLSTM + Attention")
        print("  - TCN: Captures local temporal patterns in each modality")
        print("  - BiLSTM: Captures long-range dependencies")
        print("  - Attention: Focuses on important time steps")
        print("Dropout Strategy: Strategic placement (3 locations, rate=0.25)")
        print("=" * 70)
        
        model = BalancedASLModel(
            input_size=input_size,
            num_classes=num_classes,
            hidden_size=256,
            dropout=0.25  # Conservative dropout rate
        ).to(self.device)
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"\n✅ Model built successfully!")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        print(f"   Device: {self.device}")
        print("=" * 70)
        
        self.model = model
        return model
    
    def evaluate_detailed(self, data_loader, criterion):
        """Evaluate with detailed per-class metrics"""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(data_loader.dataset)
        accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
        
        return avg_loss, accuracy, all_preds, all_labels
    
    def print_class_metrics(self, y_true, y_pred, epoch, dataset_name='Val'):
        """Print detailed per-class metrics"""
        print(f"\n{'='*70}")
        print(f"{dataset_name.upper()} SET - PER-CLASS METRICS (Epoch {epoch})")
        print(f"{'='*70}")
        
        # Get classification report
        report = classification_report(
            y_true, y_pred, 
            target_names=self.classes,
            output_dict=True,
            zero_division=0
        )
        
        # Find worst performing classes
        class_f1_scores = [(name, metrics['f1-score']) 
                          for name, metrics in report.items() 
                          if name in self.classes]
        class_f1_scores.sort(key=lambda x: x[1])
        
        print(f"\n🔴 WORST 10 CLASSES (by F1-score):")
        print(f"{'Class':25s} {'Precision':>10s} {'Recall':>10s} {'F1-Score':>10s} {'Support':>10s}")
        print("-" * 70)
        
        for class_name, _ in class_f1_scores[:10]:
            if class_name in self.classes:
                metrics = report[class_name]
                print(f"{class_name:25s} {metrics['precision']:10.3f} {metrics['recall']:10.3f} "
                      f"{metrics['f1-score']:10.3f} {metrics['support']:10.0f}")
        
        print(f"\n🟢 BEST 10 CLASSES (by F1-score):")
        print(f"{'Class':25s} {'Precision':>10s} {'Recall':>10s} {'F1-Score':>10s} {'Support':>10s}")
        print("-" * 70)
        
        for class_name, _ in class_f1_scores[-10:]:
            if class_name in self.classes:
                metrics = report[class_name]
                print(f"{class_name:25s} {metrics['precision']:10.3f} {metrics['recall']:10.3f} "
                      f"{metrics['f1-score']:10.3f} {metrics['support']:10.0f}")
        
        print(f"\n📊 OVERALL METRICS:")
        print(f"  Macro Avg    - Precision: {report['macro avg']['precision']:.4f}, "
              f"Recall: {report['macro avg']['recall']:.4f}, "
              f"F1: {report['macro avg']['f1-score']:.4f}")
        print(f"  Weighted Avg - Precision: {report['weighted avg']['precision']:.4f}, "
              f"Recall: {report['weighted avg']['recall']:.4f}, "
              f"F1: {report['weighted avg']['f1-score']:.4f}")
        print("=" * 70)
        
        return report
    
    def plot_confusion_matrix(self, y_true, y_pred, epoch):
        """Plot confusion matrix for top confused classes"""
        cm = confusion_matrix(y_true, y_pred)
        
        # Find most confused pairs
        confused_pairs = []
        for i in range(len(self.classes)):
            for j in range(len(self.classes)):
                if i != j and cm[i, j] > 0:
                    confused_pairs.append((i, j, cm[i, j]))
        
        confused_pairs.sort(key=lambda x: x[2], reverse=True)
        
        print(f"\n🔀 TOP 15 CONFUSED CLASS PAIRS:")
        print(f"{'True Class':20s} -> {'Predicted Class':20s} {'Count':>8s}")
        print("-" * 55)
        for true_idx, pred_idx, count in confused_pairs[:15]:
            print(f"{self.classes[true_idx]:20s} -> {self.classes[pred_idx]:20s} {count:8.0f}")
        
        # Save full confusion matrix for top 50 classes
        if len(self.classes) > 50:
            # Select top 50 most frequent classes in validation
            from collections import Counter
            class_counts = Counter(y_true)
            top_classes_idx = [idx for idx, _ in class_counts.most_common(50)]
            top_classes = [self.classes[i] for i in top_classes_idx]
            
            # Filter confusion matrix
            cm_subset = cm[np.ix_(top_classes_idx, top_classes_idx)]
        else:
            top_classes = self.classes
            cm_subset = cm
        
        plt.figure(figsize=(20, 18))
        sns.heatmap(cm_subset, annot=False, fmt='d', cmap='Blues',
                    xticklabels=top_classes, yticklabels=top_classes,
                    cbar_kws={'label': 'Count'})
        plt.title(f'Confusion Matrix (Epoch {epoch})', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted', fontsize=12)
        plt.ylabel('True', fontsize=12)
        plt.xticks(rotation=90, fontsize=8)
        plt.yticks(rotation=0, fontsize=8)
        plt.tight_layout()
        plt.savefig(f'models/{self.model_name}_confusion_matrix_epoch_{epoch}.png', dpi=150)
        plt.close()
        print(f"\n✅ Confusion matrix saved")
    
    def train(self, X_train, y_train, X_val, y_val, 
              epochs=150, batch_size=16, learning_rate=0.001):  # Reduced batch_size from 32 to 16
        """Train with detailed monitoring - MEMORY OPTIMIZED"""
        print("\n" + "=" * 70)
        print("STARTING TRAINING WITH DETAILED MONITORING")
        print("=" * 70)
        print(f"⚠️  MEMORY OPTIMIZED: batch_size={batch_size}, hidden_size=192")
        print("=" * 70)
        
        # Clear GPU cache before training
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"🧹 GPU cache cleared")
        
        # Datasets with augmentation
        train_dataset = ASLDataset(X_train, y_train, len(self.classes), augment=True)
        val_dataset = ASLDataset(X_val, y_val, len(self.classes), augment=False)
        
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=0, pin_memory=True if torch.cuda.is_available() else False
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=0, pin_memory=True if torch.cuda.is_available() else False
        )
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        
        optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=learning_rate,
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler - warm up then cosine decay
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=learning_rate,
            epochs=epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.1,  # 10% warmup
            anneal_strategy='cos',
            div_factor=25.0,
            final_div_factor=1000.0
        )
        
        os.makedirs('models', exist_ok=True)
        
        history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'lr': [], 'best_classes': [], 'worst_classes': []
        }
        
        best_val_acc = 0.0
        patience_counter = 0
        patience = 25
        
        print("\n🚀 Training started...\n")
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
            for inputs, labels in pbar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                
                # Use gradient checkpointing to save memory (optional but helpful)
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                scheduler.step()
                
                train_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                train_correct += (predicted == labels).sum().item()
                train_total += labels.size(0)
                
                # Clear intermediate tensors to save memory
                del outputs, loss
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                pbar.set_postfix({
                    'loss': f'{train_loss/train_total:.4f}',
                    'acc': f'{100.*train_correct/train_total:.2f}%'
                })
            
            train_loss /= len(train_dataset)
            train_acc = train_correct / train_total
            
            # Validation phase with detailed metrics
            val_loss, val_acc, val_preds, val_labels = self.evaluate_detailed(val_loader, criterion)
            
            current_lr = optimizer.param_groups[0]['lr']
            
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['lr'].append(current_lr)
            
            print(f"\n{'='*70}")
            print(f"EPOCH {epoch+1}/{epochs} SUMMARY")
            print(f"{'='*70}")
            print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f} ({train_acc*100:.2f}%)")
            print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f} ({val_acc*100:.2f}%)")
            print(f"LR: {current_lr:.2e}")
            
            # Detailed metrics every 5 epochs or when best model
            if (epoch + 1) % 5 == 0 or val_acc > best_val_acc:
                report = self.print_class_metrics(val_labels, val_preds, epoch+1)
                
                # Save confusion matrix every 10 epochs
                if (epoch + 1) % 10 == 0:
                    self.plot_confusion_matrix(val_labels, val_preds, epoch+1)
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss
                }, f'models/{self.model_name}_best.pth')
                print(f"\n✅ BEST MODEL SAVED! Val Acc: {val_acc:.4f} ({val_acc*100:.2f}%)")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\n⚠️  Early stopping at epoch {epoch+1}")
                    print(f"Best validation accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
                    break
            
            print("=" * 70)
        
        # Final evaluation with full metrics
        print("\n" + "=" * 70)
        print("FINAL EVALUATION ON VALIDATION SET")
        print("=" * 70)
        
        # Load best model
        checkpoint = torch.load(f'models/{self.model_name}_best.pth')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        val_loss, val_acc, val_preds, val_labels = self.evaluate_detailed(val_loader, criterion)
        report = self.print_class_metrics(val_labels, val_preds, 'Final')
        self.plot_confusion_matrix(val_labels, val_preds, 'final')
        
        # Save final artifacts
        torch.save({'model_state_dict': self.model.state_dict()}, 
                  f'models/{self.model_name}_final.pth')
        with open(f'models/{self.model_name}_classes.pkl', 'wb') as f:
            pickle.dump(self.classes, f)
        with open(f'models/{self.model_name}_label_map.json', 'w') as f:
            json.dump(self.label_map, f, indent=2)
        with open(f'models/{self.model_name}_history.pkl', 'wb') as f:
            pickle.dump(history, f)
        with open(f'models/{self.model_name}_final_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        print("\n" + "=" * 70)
        print("TRAINING COMPLETE!")
        print("=" * 70)
        print(f"🏆 Best Validation Accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
        print("=" * 70)
        
        return history
    
    def plot_training_history(self, history):
        """Plot training metrics"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        epochs_range = range(1, len(history['train_acc']) + 1)
        
        # Accuracy
        axes[0].plot(epochs_range, history['train_acc'], label='Train', linewidth=2, marker='o', markersize=3)
        axes[0].plot(epochs_range, history['val_acc'], label='Val', linewidth=2, marker='s', markersize=3)
        axes[0].set_title('Accuracy', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Loss
        axes[1].plot(epochs_range, history['train_loss'], label='Train', linewidth=2, marker='o', markersize=3)
        axes[1].plot(epochs_range, history['val_loss'], label='Val', linewidth=2, marker='s', markersize=3)
        axes[1].set_title('Loss', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Learning Rate
        axes[2].plot(epochs_range, history['lr'], linewidth=2, color='green', marker='d', markersize=3)
        axes[2].set_title('Learning Rate', fontsize=14, fontweight='bold')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('LR')
        axes[2].set_yscale('log')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'models/{self.model_name}_training_history.png', dpi=300)
        print(f"✅ Training plots saved")
        plt.close()


def main():
    print("\n" + "=" * 70)
    print("ASL RECOGNITION - BALANCED TRAINING")
    print("=" * 70)
    
    trainer = ASLModelTrainer(model_name='asl_balanced_model')
    X_train, y_train, X_val, y_val = trainer.load_data()
    
    if X_train is None:
        return
    
    trainer.build_model(X_train.shape[2], len(trainer.classes))
    
    history = trainer.train(
        X_train, y_train, X_val, y_val,
        epochs=150,
        batch_size=16,  # Reduced from 32 to 16 for memory
        learning_rate=0.001
    )
    
    trainer.plot_training_history(history)
    
    best_val_acc = max(history['val_acc'])
    print(f"\n🏆 Best Validation Accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
    print("=" * 70)


if __name__ == "__main__":
    main()