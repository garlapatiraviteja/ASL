"""
MediaPipe + LSTM Pipeline for WLASL-100 Sign Language Recognition
Based on Nick Nochnack's Action Detection architecture

Architecture:
1. MediaPipe Holistic extracts keypoints (hands, pose, face)
2. LSTM processes temporal sequences
3. Dense classifier predicts ASL sign

Dataset: WLASL-100 (942 videos, 100 classes)
"""

import cv2
import numpy as np
import mediapipe as mp
import json
from pathlib import Path
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

# Deep Learning
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Training configuration"""
    
    # Paths
    DATASET_DIR = 'wlasl_100_best'
    VIDEOS_DIR = 'videos'
    DATA_PATH = 'MP_Data'  # Processed MediaPipe data
    MODEL_PATH = 'models'
    
    # MediaPipe settings
    SEQUENCE_LENGTH = 30  # Number of frames per video
    
    # Model hyperparameters
    LSTM_UNITS = [128, 64, 32]  # LSTM layers
    DROPOUT_RATE = 0.5
    LEARNING_RATE = 0.001
    
    # Training settings
    BATCH_SIZE = 32
    EPOCHS = 200
    PATIENCE = 15  # Early stopping patience
    
    # Create directories
    os.makedirs(DATA_PATH, exist_ok=True)
    os.makedirs(MODEL_PATH, exist_ok=True)


# ============================================================================
# STEP 1: MEDIAPIPE FEATURE EXTRACTION
# ============================================================================

class MediaPipeExtractor:
    """Extract keypoints using MediaPipe Holistic"""
    
    def __init__(self):
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        
    def mediapipe_detection(self, image, model):
        """Run MediaPipe detection"""
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = model.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image, results
    
    def extract_keypoints(self, results):
        """
        Extract keypoints from MediaPipe results
        Returns flattened array of all keypoints
        """
        # Pose: 33 landmarks × 4 (x, y, z, visibility) = 132 values
        pose = np.array([[res.x, res.y, res.z, res.visibility] 
                        for res in results.pose_landmarks.landmark]).flatten() \
               if results.pose_landmarks else np.zeros(33*4)
        
        # Face: 468 landmarks × 3 (x, y, z) = 1404 values
        face = np.array([[res.x, res.y, res.z] 
                        for res in results.face_landmarks.landmark]).flatten() \
               if results.face_landmarks else np.zeros(468*3)
        
        # Left hand: 21 landmarks × 3 = 63 values
        lh = np.array([[res.x, res.y, res.z] 
                      for res in results.left_hand_landmarks.landmark]).flatten() \
             if results.left_hand_landmarks else np.zeros(21*3)
        
        # Right hand: 21 landmarks × 3 = 63 values
        rh = np.array([[res.x, res.y, res.z] 
                      for res in results.right_hand_landmarks.landmark]).flatten() \
             if results.right_hand_landmarks else np.zeros(21*3)
        
        # Total: 132 + 1404 + 63 + 63 = 1662 features per frame
        return np.concatenate([pose, face, lh, rh])
    
    def draw_styled_landmarks(self, image, results):
        """Draw landmarks on image for visualization"""
        # Pose
        self.mp_drawing.draw_landmarks(
            image, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS,
            self.mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
            self.mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
        )
        # Face
        self.mp_drawing.draw_landmarks(
            image, results.face_landmarks, self.mp_holistic.FACEMESH_CONTOURS,
            self.mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
            self.mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
        )
        # Hands
        self.mp_drawing.draw_landmarks(
            image, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
            self.mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
            self.mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
        )
        self.mp_drawing.draw_landmarks(
            image, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
            self.mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
            self.mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
        )
        
        return image
    
    def process_video(self, video_path, sequence_length=30):
        """
        Process video and extract keypoint sequences
        
        Args:
            video_path: Path to video file
            sequence_length: Number of frames to sample
            
        Returns:
            numpy array of shape (sequence_length, 1662)
        """
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            return None
        
        # Get total frames
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            cap.release()
            return None
        
        # Sample frame indices uniformly
        frame_indices = np.linspace(0, total_frames - 1, sequence_length, dtype=int)
        
        sequence = []
        
        with self.mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as holistic:
            
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if not ret:
                    # If frame read fails, use zeros
                    sequence.append(np.zeros(1662))
                    continue
                
                # MediaPipe detection
                image, results = self.mediapipe_detection(frame, holistic)     
                
                # Extract keypoints
                keypoints = self.extract_keypoints(results)
                sequence.append(keypoints)
        
        cap.release()
        
        # Ensure we have exactly sequence_length frames
        if len(sequence) < sequence_length:
            # Pad with zeros if needed
            while len(sequence) < sequence_length:
                sequence.append(np.zeros(1662))
        
        return np.array(sequence)


# ============================================================================
# STEP 2: DATASET PREPARATION
# ============================================================================

class WLASLDatasetProcessor:
    """Process WLASL dataset and extract MediaPipe features"""
    
    def __init__(self, config):
        self.config = config
        self.extractor = MediaPipeExtractor()
        
        # Load class mapping
        with open(Path(config.DATASET_DIR) / 'class_mapping.json', 'r') as f:
            mapping = json.load(f)
        
        self.class_to_idx = mapping['class_to_idx']
        self.idx_to_class = {int(k): v for k, v in mapping['idx_to_class'].items()}
        self.num_classes = mapping['num_classes']
        
        print(f"✅ Loaded {self.num_classes} classes")
    
    def load_split_data(self, split='train'):
        """Load train/val/test split"""
        split_path = Path(self.config.DATASET_DIR) / 'annotations' / f'{split}.json'
        
        with open(split_path, 'r') as f:
            data = json.load(f)
        
        return data
    
    def process_dataset(self, force_reprocess=False):
        """
        Process all videos and save MediaPipe features
        
        This extracts keypoints for all videos and saves them as .npy files
        """
        print("\n" + "="*70)
        print("🎬 PROCESSING VIDEOS WITH MEDIAPIPE")
        print("="*70)
        
        for split in ['train', 'val', 'test']:
            print(f"\n📁 Processing {split.upper()} split...")
            
            split_data = self.load_split_data(split)
            split_path = Path(self.config.DATA_PATH) / split
            split_path.mkdir(exist_ok=True, parents=True)
            
            processed = 0
            skipped = 0
            failed = 0
            
            for video_info in tqdm(split_data, desc=f"Processing {split}"):
                video_id = video_info['video_id']
                class_id = video_info['class_id']
                class_name = video_info['class_name']
                
                # Create class folder
                class_folder = split_path / f"{class_id}_{class_name}"
                class_folder.mkdir(exist_ok=True)
                
                # Check if already processed
                output_path = class_folder / f"{video_id}.npy"
                if output_path.exists() and not force_reprocess:
                    skipped += 1
                    continue
                
                # Find video file
                video_path = None
                for ext in ['.mp4', '.mkv', '.avi', '.webm', '.mov', '']:
                    path = Path(self.config.VIDEOS_DIR) / f"{video_id}{ext}"
                    if path.exists():
                        video_path = path
                        break
                
                if not video_path:
                    failed += 1
                    continue
                
                # Extract keypoints
                try:
                    sequence = self.extractor.process_video(
                        video_path, 
                        sequence_length=self.config.SEQUENCE_LENGTH
                    )
                    
                    if sequence is not None:
                        np.save(output_path, sequence)
                        processed += 1
                    else:
                        failed += 1
                        
                except Exception as e:
                    failed += 1
            
            print(f"\n   ✅ Processed: {processed}")
            print(f"   ⏭️  Skipped (already done): {skipped}")
            print(f"   ❌ Failed: {failed}")
    
    def load_processed_data(self, split='train'):
        """Load processed MediaPipe sequences"""
        split_path = Path(self.config.DATA_PATH) / split
        
        X = []
        y = []
        
        for class_folder in sorted(split_path.iterdir()):
            if not class_folder.is_dir():
                continue
            
            # Extract class_id from folder name (format: "0_hello")
            class_id = int(class_folder.name.split('_')[0])
            
            # Load all sequences for this class
            for npy_file in class_folder.glob('*.npy'):
                try:
                    sequence = np.load(npy_file)
                    X.append(sequence)
                    y.append(class_id)
                except:
                    continue
        
        return np.array(X), np.array(y)


# ============================================================================
# STEP 3: BUILD LSTM MODEL
# ============================================================================

def build_lstm_model(sequence_length, num_features, num_classes, config):
    """
    Build LSTM model for sign language recognition
    
    Architecture (Nick Nochnack style):
    - LSTM layers with decreasing units
    - Dropout for regularization
    - Dense output layer with softmax
    """
    model = Sequential([
        # First LSTM layer
        LSTM(config.LSTM_UNITS[0], return_sequences=True, activation='relu',
             input_shape=(sequence_length, num_features)),
        Dropout(config.DROPOUT_RATE),
        
        # Second LSTM layer
        LSTM(config.LSTM_UNITS[1], return_sequences=True, activation='relu'),
        Dropout(config.DROPOUT_RATE),
        
        # Third LSTM layer
        LSTM(config.LSTM_UNITS[2], return_sequences=False, activation='relu'),
        Dropout(config.DROPOUT_RATE),
        
        # Dense layers
        Dense(64, activation='relu'),
        Dropout(config.DROPOUT_RATE),
        
        Dense(32, activation='relu'),
        
        # Output layer
        Dense(num_classes, activation='softmax')
    ])
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['categorical_accuracy']
    )
    
    return model


# ============================================================================
# STEP 4: TRAINING
# ============================================================================

class Trainer:
    """Train LSTM model on WLASL dataset"""
    
    def __init__(self, config):
        self.config = config
        self.processor = WLASLDatasetProcessor(config)
        self.history = None
        self.model = None
    
    def prepare_data(self):
        """Load and prepare training data"""
        print("\n" + "="*70)
        print("📊 LOADING PROCESSED DATA")
        print("="*70)
        
        # Load splits
        print("\n📥 Loading training data...")
        X_train, y_train = self.processor.load_processed_data('train')
        print(f"   Train: {X_train.shape[0]} sequences")
        
        print("📥 Loading validation data...")
        X_val, y_val = self.processor.load_processed_data('val')
        print(f"   Val: {X_val.shape[0]} sequences")
        
        print("📥 Loading test data...")
        X_test, y_test = self.processor.load_processed_data('test')
        print(f"   Test: {X_test.shape[0]} sequences")
        
        # Convert labels to categorical
        y_train_cat = to_categorical(y_train, num_classes=self.processor.num_classes)
        y_val_cat = to_categorical(y_val, num_classes=self.processor.num_classes)
        y_test_cat = to_categorical(y_test, num_classes=self.processor.num_classes)
        
        print(f"\n📐 Data shapes:")
        print(f"   X_train: {X_train.shape}")
        print(f"   y_train: {y_train_cat.shape}")
        print(f"   Features per frame: {X_train.shape[2]}")
        
        return (X_train, y_train_cat), (X_val, y_val_cat), (X_test, y_test_cat)
    
    def train(self):
        """Train the model"""
        # Prepare data
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = self.prepare_data()
        
        # Build model
        print("\n" + "="*70)
        print("🏗️  BUILDING MODEL")
        print("="*70)
        
        self.model = build_lstm_model(
            sequence_length=self.config.SEQUENCE_LENGTH,
            num_features=X_train.shape[2],
            num_classes=self.processor.num_classes,
            config=self.config
        )
        
        print(self.model.summary())
        
        # Callbacks
        log_dir = os.path.join('Logs')
        tb_callback = TensorBoard(log_dir=log_dir)
        
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=self.config.PATIENCE,
            restore_best_weights=True,
            verbose=1
        )
        
        checkpoint = ModelCheckpoint(
            os.path.join(self.config.MODEL_PATH, 'best_model.h5'),
            monitor='val_categorical_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )
        
        # Train
        print("\n" + "="*70)
        print("🚀 STARTING TRAINING")
        print("="*70)
        print(f"\nEpochs: {self.config.EPOCHS}")
        print(f"Batch size: {self.config.BATCH_SIZE}")
        print(f"Learning rate: {self.config.LEARNING_RATE}")
        print(f"Early stopping patience: {self.config.PATIENCE}")
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.config.EPOCHS,
            batch_size=self.config.BATCH_SIZE,
            callbacks=[tb_callback, early_stop, checkpoint],
            verbose=1
        )
        
        # Evaluate on test set
        print("\n" + "="*70)
        print("📈 EVALUATING ON TEST SET")
        print("="*70)
        
        test_loss, test_acc = self.model.evaluate(X_test, y_test)
        print(f"\n✅ Test Loss: {test_loss:.4f}")
        print(f"✅ Test Accuracy: {test_acc*100:.2f}%")
        
        # Save final model
        final_model_path = os.path.join(self.config.MODEL_PATH, 'final_model.h5')
        self.model.save(final_model_path)
        print(f"\n💾 Model saved to: {final_model_path}")
        
        return self.history, test_acc
    
    def plot_training_history(self):
        """Plot training curves"""
        if not self.history:
            print("❌ No training history available")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Accuracy
        ax1.plot(self.history.history['categorical_accuracy'], label='Train Accuracy', linewidth=2)
        ax1.plot(self.history.history['val_categorical_accuracy'], label='Val Accuracy', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Accuracy', fontsize=12)
        ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Loss
        ax2.plot(self.history.history['loss'], label='Train Loss', linewidth=2)
        ax2.plot(self.history.history['val_loss'], label='Val Loss', linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Loss', fontsize=12)
        ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
        print("\n✅ Training history saved as 'training_history.png'")
        plt.show()


# ============================================================================
# STEP 5: REAL-TIME INFERENCE
# ============================================================================

class RealtimePredictor:
    """Real-time sign language prediction using webcam"""
    
    def __init__(self, model_path, class_mapping):
        self.model = keras.models.load_model(model_path)
        self.idx_to_class = class_mapping
        self.extractor = MediaPipeExtractor()
        
        self.sequence = []
        self.predictions = []
        self.threshold = 0.7  # Confidence threshold
        
        print(f"✅ Model loaded from: {model_path}")
        print(f"✅ Number of classes: {len(self.idx_to_class)}")
    
    def predict(self, sequence_length=30):
        """Run real-time prediction on webcam"""
        cap = cv2.VideoCapture(0)
        
        with self.extractor.mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as holistic:
            
            while cap.isOpened():
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # MediaPipe detection
                image, results = self.extractor.mediapipe_detection(frame, holistic)
                
                # Draw landmarks
                image = self.extractor.draw_styled_landmarks(image, results)
                
                # Extract keypoints
                keypoints = self.extractor.extract_keypoints(results)
                self.sequence.append(keypoints)
                self.sequence = self.sequence[-sequence_length:]
                
                # Predict when we have enough frames
                if len(self.sequence) == sequence_length:
                    res = self.model.predict(np.expand_dims(self.sequence, axis=0), verbose=0)[0]
                    predicted_class = np.argmax(res)
                    confidence = res[predicted_class]
                    
                    self.predictions.append(predicted_class)
                    
                    # Display prediction
                    if confidence > self.threshold:
                        class_name = self.idx_to_class[predicted_class]
                        
                        # Draw prediction on frame
                        cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
                        cv2.putText(image, f'{class_name} ({confidence:.2f})', 
                                  (3, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                # Display frame
                cv2.imshow('ASL Recognition', image)
                
                # Break on 'q' key
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
            
            cap.release()
            cv2.destroyAllWindows()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main training pipeline"""
    
    print("="*70)
    print("🎯 MEDIAPIPE + LSTM FOR WLASL-100 SIGN LANGUAGE RECOGNITION")
    print("="*70)
    
    # Initialize config
    config = Config()
    
    # Step 1: Process videos (extract MediaPipe keypoints)
    print("\n" + "="*70)
    print("STEP 1: EXTRACT MEDIAPIPE KEYPOINTS FROM VIDEOS")
    print("="*70)
    
    processor = WLASLDatasetProcessor(config)
    
    choice = input("\nDo you want to (re)process all videos? (y/n): ").lower()
    if choice == 'y':
        processor.process_dataset(force_reprocess=True)
    else:
        print("✅ Skipping video processing (will use existing processed data)")
    
    # Step 2: Train model
    print("\n" + "="*70)
    print("STEP 2: TRAIN LSTM MODEL")
    print("="*70)
    
    trainer = Trainer(config)
    history, test_acc = trainer.train()
    
    # Step 3: Plot results
    trainer.plot_training_history()
    
    # Step 4: Real-time inference
    print("\n" + "="*70)
    print("STEP 3: REAL-TIME INFERENCE")
    print("="*70)
    
    choice = input("\nDo you want to test real-time prediction? (y/n): ").lower()
    if choice == 'y':
        predictor = RealtimePredictor(
            model_path=os.path.join(config.MODEL_PATH, 'best_model.h5'),
            class_mapping=processor.idx_to_class
        )
        print("\n🎥 Starting webcam... Press 'q' to quit")
        predictor.predict(sequence_length=config.SEQUENCE_LENGTH)
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70)
    print(f"\n📊 Final Results:")
    print(f"   Test Accuracy: {test_acc*100:.2f}%")
    print(f"   Model saved: {config.MODEL_PATH}/best_model.h5")
    print(f"   Training history: training_history.png")


if __name__ == "__main__":
    main()