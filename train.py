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

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Check GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if torch.cuda.is_available():
    print(f"✅ GPU Available: {torch.cuda.get_device_name(0)}")
    print(f"   CUDA Version: {torch.version.cuda}")
else:
    print("⚠️  No GPU detected. Training will use CPU (slower)")


class ASLDataset(Dataset):
    """PyTorch Dataset for ASL sequences"""
    
    def __init__(self, sequences, labels):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


class ASLModel(nn.Module):
    """
    PyTorch LSTM model for ASL recognition
    Architecture: Bidirectional LSTM with Attention
    """
    
    def __init__(self, input_size, num_classes, hidden_size=128, num_layers=3, dropout=0.4):
        super(ASLModel, self).__init__()
        
        # First Bidirectional LSTM
        self.lstm1 = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0
        )
        self.bn1 = nn.BatchNorm1d(hidden_size * 2)
        self.dropout1 = nn.Dropout(0.3)
        
        # Second Bidirectional LSTM
        self.lstm2 = nn.LSTM(
            input_size=hidden_size * 2,
            hidden_size=hidden_size * 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0
        )
        self.bn2 = nn.BatchNorm1d(hidden_size * 4)
        self.dropout2 = nn.Dropout(0.4)
        
        # Third LSTM (unidirectional)
        self.lstm3 = nn.LSTM(
            input_size=hidden_size * 4,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
            dropout=0
        )
        self.bn3 = nn.BatchNorm1d(hidden_size)
        self.dropout3 = nn.Dropout(0.4)
        
        # Dense layers
        self.fc1 = nn.Linear(hidden_size, 256)
        self.bn4 = nn.BatchNorm1d(256)
        self.dropout4 = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(256, 128)
        self.dropout5 = nn.Dropout(0.4)
        
        # Output layer
        self.fc3 = nn.Linear(128, num_classes)
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        
        # First LSTM
        x, _ = self.lstm1(x)
        # Transpose for batch norm: (batch, features, seq_len)
        x = x.transpose(1, 2)
        x = self.bn1(x)
        x = x.transpose(1, 2)
        x = self.dropout1(x)
        
        # Second LSTM
        x, _ = self.lstm2(x)
        x = x.transpose(1, 2)
        x = self.bn2(x)
        x = x.transpose(1, 2)
        x = self.dropout2(x)
        
        # Third LSTM
        x, _ = self.lstm3(x)
        # Take last hidden state
        x = x[:, -1, :]
        x = self.bn3(x)
        x = self.dropout3(x)
        
        # Dense layers
        x = self.relu(self.fc1(x))
        x = self.bn4(x)
        x = self.dropout4(x)
        
        x = self.relu(self.fc2(x))
        x = self.dropout5(x)
        
        # Output
        x = self.fc3(x)
        
        return x


class ASLModelTrainer:
    """ASL Sign Language Recognition Model Trainer - PyTorch Version"""
    
    def __init__(self, data_dir='augmented_data', model_name='asl_realtime_model'):
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
            print("Make sure you ran data_augmentation.py first!")
            return None, None, None, None
        
        # Get class names
        self.classes = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
        self.label_map = {label: i for i, label in enumerate(self.classes)}
        
        print(f"\n📊 Dataset Information:")
        print(f"  Classes: {len(self.classes)}")
        print(f"  First 10: {self.classes[:10]}")
        
        # Load training data
        print("\n📥 Loading training data...")
        X_train, y_train = self._load_from_directory(train_dir)
        
        # Load validation data
        print("\n📥 Loading validation data...")
        X_val, y_val = self._load_from_directory(val_dir)
        
        print("\n" + "=" * 70)
        print("DATA LOADING COMPLETE!")
        print("=" * 70)
        print(f"✅ Training samples: {len(X_train)}")
        print(f"✅ Validation samples: {len(X_val)}")
        print(f"✅ Sequence length: {X_train.shape[1]} frames")
        print(f"✅ Features per frame: {X_train.shape[2]}")
        print(f"   - Pose: 33 landmarks × 4 = 132")
        print(f"   - Face: 468 landmarks × 3 = 1,404")
        print(f"   - Left Hand: 21 landmarks × 3 = 63")
        print(f"   - Right Hand: 21 landmarks × 3 = 63")
        print(f"   - Total: {X_train.shape[2]} features")
        print(f"✅ Classes: {len(self.classes)}")
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
                    
                    # Validate shape
                    if sequence.shape[0] != 30 or sequence.shape[1] != 1662:
                        print(f"\n  ⚠️  Skipping {npy_file.name}: Invalid shape {sequence.shape}")
                        continue
                    
                    # Check for NaN/Inf
                    if np.isnan(sequence).any() or np.isinf(sequence).any():
                        print(f"\n  ⚠️  Skipping {npy_file.name}: Contains NaN/Inf")
                        continue
                    
                    sequences.append(sequence)
                    labels.append(class_label)
                    
                except Exception as e:
                    print(f"\n  ❌ Error loading {npy_file}: {e}")
                    continue
        
        print()  # New line after progress
        
        X = np.array(sequences)
        y = np.array(labels)
        
        return X, y
    
    def build_model(self, input_size, num_classes):
        """Build PyTorch LSTM model"""
        print("\n" + "=" * 70)
        print("BUILDING ASL RECOGNITION MODEL (PyTorch)")
        print("=" * 70)
        print("Architecture: Bidirectional LSTM with BatchNorm")
        print("=" * 70)
        
        model = ASLModel(
            input_size=input_size,
            num_classes=num_classes,
            hidden_size=128,
            num_layers=3,
            dropout=0.4
        ).to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print("\n📊 Model Architecture:")
        print(model)
        print(f"\n✅ Model built successfully!")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        print(f"   Input size: {input_size}")
        print(f"   Output classes: {num_classes}")
        print(f"   Device: {self.device}")
        print("=" * 70)
        
        self.model = model
        return model
    
    def train(self, X_train, y_train, X_val, y_val, 
              epochs=150, batch_size=32, learning_rate=0.001):
        """Train the ASL recognition model"""
        print("\n" + "=" * 70)
        print("STARTING MODEL TRAINING")
        print("=" * 70)
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Epochs: {epochs}")
        print(f"Batch size: {batch_size}")
        print(f"Learning rate: {learning_rate}")
        print(f"Steps per epoch: {len(X_train) // batch_size}")
        print("=" * 70)
        
        # Create datasets and dataloaders
        train_dataset = ASLDataset(X_train, y_train)
        val_dataset = ASLDataset(X_val, y_val)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=0,
            pin_memory=True if torch.cuda.is_available() else False
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=0,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, 
                              betas=(0.9, 0.999), weight_decay=0.001)
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=8, 
            min_lr=1e-7, verbose=True
        )
        
        # Create directories
        os.makedirs('models', exist_ok=True)
        
        # Training history
        history = {
            'train_loss': [], 'train_acc': [], 'train_top5': [],
            'val_loss': [], 'val_acc': [], 'val_top5': [],
            'lr': []
        }
        
        best_val_acc = 0.0
        patience_counter = 0
        patience = 20
        
        print("\n🚀 Training started...\n")
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_top5_correct = 0
            train_total = 0
            
            pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
            for inputs, labels in pbar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # Forward pass
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Statistics
                train_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                train_correct += (predicted == labels).sum().item()
                train_total += labels.size(0)
                
                # Top-5 accuracy
                _, top5_pred = outputs.topk(5, 1, True, True)
                top5_pred = top5_pred.t()
                correct_top5 = top5_pred.eq(labels.view(1, -1).expand_as(top5_pred))
                train_top5_correct += correct_top5.sum().item()
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100.*train_correct/train_total:.2f}%'
                })
            
            train_loss = train_loss / len(train_dataset)
            train_acc = train_correct / train_total
            train_top5_acc = train_top5_correct / train_total
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_top5_correct = 0
            val_total = 0
            
            with torch.no_grad():
                pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Val]  ')
                for inputs, labels in pbar:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    
                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs.data, 1)
                    val_correct += (predicted == labels).sum().item()
                    val_total += labels.size(0)
                    
                    # Top-5 accuracy
                    _, top5_pred = outputs.topk(5, 1, True, True)
                    top5_pred = top5_pred.t()
                    correct_top5 = top5_pred.eq(labels.view(1, -1).expand_as(top5_pred))
                    val_top5_correct += correct_top5.sum().item()
                    
                    pbar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'acc': f'{100.*val_correct/val_total:.2f}%'
                    })
            
            val_loss = val_loss / len(val_dataset)
            val_acc = val_correct / val_total
            val_top5_acc = val_top5_correct / val_total
            
            # Update learning rate
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            
            # Save history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['train_top5'].append(train_top5_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['val_top5'].append(val_top5_acc)
            history['lr'].append(current_lr)
            
            # Print epoch summary
            print(f"\nEpoch {epoch+1}/{epochs}:")
            print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f} ({train_acc*100:.2f}%), Top-5: {train_top5_acc:.4f}")
            print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f} ({val_acc*100:.2f}%), Top-5: {val_top5_acc:.4f}")
            print(f"  LR: {current_lr:.2e}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss,
                }, f'models/{self.model_name}_best.pth')
                print(f"  ✅ Best model saved! Val Acc: {val_acc:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\n⚠️  Early stopping triggered after {epoch+1} epochs")
                    break
            
            print("-" * 70)
        
        print("\n" + "=" * 70)
        print("TRAINING COMPLETE!")
        print("=" * 70)
        
        # Save final model
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, f'models/{self.model_name}_final.pth')
        print(f"✅ Final model saved: models/{self.model_name}_final.pth")
        print(f"✅ Best model saved: models/{self.model_name}_best.pth")
        
        # Save class mapping
        with open(f'models/{self.model_name}_classes.pkl', 'wb') as f:
            pickle.dump(self.classes, f)
        print(f"✅ Classes saved: models/{self.model_name}_classes.pkl")
        
        # Save label mapping
        with open(f'models/{self.model_name}_label_map.json', 'w') as f:
            json.dump(self.label_map, f, indent=2)
        print(f"✅ Label map saved: models/{self.model_name}_label_map.json")
        
        # Save training history
        with open(f'models/{self.model_name}_history.pkl', 'wb') as f:
            pickle.dump(history, f)
        print(f"✅ Training history saved: models/{self.model_name}_history.pkl")
        
        print("=" * 70)
        
        return history
    
    def plot_training_history(self, history):
        """Plot training metrics"""
        print("\n📊 Generating training plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        epochs_range = range(1, len(history['train_acc']) + 1)
        
        # Accuracy
        axes[0, 0].plot(epochs_range, history['train_acc'], 
                       label='Train Accuracy', linewidth=2)
        axes[0, 0].plot(epochs_range, history['val_acc'], 
                       label='Val Accuracy', linewidth=2)
        axes[0, 0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend(loc='lower right')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Loss
        axes[0, 1].plot(epochs_range, history['train_loss'], 
                       label='Train Loss', linewidth=2)
        axes[0, 1].plot(epochs_range, history['val_loss'], 
                       label='Val Loss', linewidth=2)
        axes[0, 1].set_title('Model Loss', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend(loc='upper right')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Top-5 Accuracy
        axes[1, 0].plot(epochs_range, history['train_top5'], 
                       label='Train Top-5', linewidth=2)
        axes[1, 0].plot(epochs_range, history['val_top5'], 
                       label='Val Top-5', linewidth=2)
        axes[1, 0].set_title('Top-5 Accuracy', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].legend(loc='lower right')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Learning Rate
        axes[1, 1].plot(epochs_range, history['lr'], linewidth=2, color='green')
        axes[1, 1].set_title('Learning Rate', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(f'ASL Model Training History (PyTorch) - {self.model_name}', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        plot_path = f'models/{self.model_name}_training_history.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✅ Training plots saved: {plot_path}")
        plt.close()
    
    def evaluate(self, X_test, y_test, batch_size=32):
        """Evaluate model on test set"""
        print("\n" + "=" * 70)
        print("EVALUATING MODEL")
        print("=" * 70)
        
        test_dataset = ASLDataset(X_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        self.model.eval()
        test_loss = 0.0
        test_correct = 0
        test_top5_correct = 0
        test_total = 0
        
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for inputs, labels in tqdm(test_loader, desc='Testing'):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                
                test_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                test_correct += (predicted == labels).sum().item()
                test_total += labels.size(0)
                
                # Top-5 accuracy
                _, top5_pred = outputs.topk(5, 1, True, True)
                top5_pred = top5_pred.t()
                correct_top5 = top5_pred.eq(labels.view(1, -1).expand_as(top5_pred))
                test_top5_correct += correct_top5.sum().item()
        
        test_loss = test_loss / len(test_dataset)
        test_acc = test_correct / test_total
        test_top5_acc = test_top5_correct / test_total
        
        print("\n📊 Test Results:")
        print(f"  Loss: {test_loss:.4f}")
        print(f"  Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
        print(f"  Top-5 Accuracy: {test_top5_acc:.4f} ({test_top5_acc*100:.2f}%)")
        print("=" * 70)
        
        return test_loss, test_acc, test_top5_acc


def main():
    """Main training pipeline for ASL recognition"""
    print("\n" + "=" * 70)
    print("ASL REAL-TIME SIGN LANGUAGE RECOGNITION")
    print("Model Training Pipeline (PyTorch)")
    print("=" * 70)
    print("\n🎯 Training on WLASL Dataset")
    print("📊 100 Sign Classes")
    print("🤖 Bidirectional LSTM Architecture")
    print("👁️ Full Body Keypoints (Pose + Face + Hands)")
    print(f"🚀 Device: {device}")
    print("=" * 70)
    
    # Configuration
    DATA_DIR = 'augmented_data'
    MODEL_NAME = 'asl_realtime_model_pytorch'
    EPOCHS = 150
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    
    print(f"\n⚙️  Configuration:")
    print(f"  Data directory: {DATA_DIR}")
    print(f"  Model name: {MODEL_NAME}")
    print(f"  Max epochs: {EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Learning rate: {LEARNING_RATE}")
    
    # Initialize trainer
    trainer = ASLModelTrainer(data_dir=DATA_DIR, model_name=MODEL_NAME)
    
    # Load data
    X_train, y_train, X_val, y_val = trainer.load_data()
    
    if X_train is None:
        print("\n❌ Failed to load data. Exiting...")
        return
    
    # Build model
    input_size = X_train.shape[2]  # 1662 features
    num_classes = len(trainer.classes)
    trainer.build_model(input_size, num_classes)
    
    # Train model
    history = trainer.train(
        X_train, y_train,
        X_val, y_val,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE
    )
    
    # Plot training history
    trainer.plot_training_history(history)
    
    # Final evaluation
    trainer.evaluate(X_val, y_val)
    
    print("\n" + "=" * 70)
    print("TRAINING PIPELINE COMPLETE!")
    print("=" * 70)
    print("\n📁 Generated Files:")
    print(f"  1. models/{MODEL_NAME}_best.pth - Best model")
    print(f"  2. models/{MODEL_NAME}_final.pth - Final model")
    print(f"  3. models/{MODEL_NAME}_classes.pkl - Class labels")
    print(f"  4. models/{MODEL_NAME}_label_map.json - Label mapping")
    print(f"  5. models/{MODEL_NAME}_training_history.png - Training plots")
    print("\n🎯 Next Steps:")
    print("  1. Review training plots for overfitting")
    print("  2. Test with real-time recognition (update realtime.py for PyTorch)")
    print("=" * 70)
    
    # Print best results
    best_val_acc = max(history['val_acc'])
    best_epoch = history['val_acc'].index(best_val_acc) + 1
    print(f"\n🏆 Best Validation Accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
    print(f"   Achieved at epoch: {best_epoch}")
    print("=" * 70)


if __name__ == "__main__":
    main()