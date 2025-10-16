import cv2
import numpy as np
import mediapipe as mp
import json
import os
from pathlib import Path
import pickle

os.chdir(os.path.dirname(os.path.abspath(__file__)))


class WASLVideoPreprocessor:
    """
    Preprocess WLASL videos for training:
    - Extract keypoints using MediaPipe Holistic
    - Save sequences in format compatible with ActionDetection model
    """
    
    def __init__(self, video_dir='videos', output_dir='processed_data'):
        self.video_dir = Path(video_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize MediaPipe Holistic
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
    def extract_keypoints(self, results):
        """Extract keypoints from MediaPipe results"""
        # Pose: 33 landmarks x 4 (x, y, z, visibility)
        pose = np.array([[res.x, res.y, res.z, res.visibility] 
                        for res in results.pose_landmarks.landmark]).flatten() \
               if results.pose_landmarks else np.zeros(33*4)
        
        # Face: 468 landmarks x 3 (x, y, z)
        face = np.array([[res.x, res.y, res.z] 
                        for res in results.face_landmarks.landmark]).flatten() \
               if results.face_landmarks else np.zeros(468*3)
        
        # Left hand: 21 landmarks x 3 (x, y, z)
        lh = np.array([[res.x, res.y, res.z] 
                      for res in results.left_hand_landmarks.landmark]).flatten() \
             if results.left_hand_landmarks else np.zeros(21*3)
        
        # Right hand: 21 landmarks x 3 (x, y, z)
        rh = np.array([[res.x, res.y, res.z] 
                      for res in results.right_hand_landmarks.landmark]).flatten() \
             if results.right_hand_landmarks else np.zeros(21*3)
        
        return np.concatenate([pose, face, lh, rh])
    
    def process_video(self, video_path, frame_start=None, frame_end=None, 
                     target_frames=30):
        """
        Process a single video and extract keypoint sequences
        
        Args:
            video_path: Path to video file
            frame_start: Start frame (optional)
            frame_end: End frame (optional)
            target_frames: Target number of frames to extract
        
        Returns:
            numpy array of keypoints (frames x features)
        """
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            print(f"Error opening video: {video_path}")
            return None
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Set frame range
        if frame_start is None:
            frame_start = 0
        if frame_end is None or frame_end > total_frames:
            frame_end = total_frames
        
        # Calculate frame indices to sample
        frame_indices = np.linspace(frame_start, frame_end-1, 
                                   target_frames, dtype=int)
        
        keypoint_sequence = []
        
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                continue
            
            # Convert to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            
            # Process with MediaPipe
            results = self.holistic.process(image)
            
            # Extract keypoints
            keypoints = self.extract_keypoints(results)
            keypoint_sequence.append(keypoints)
        
        cap.release()
        
        # Pad or truncate to target_frames
        keypoint_sequence = np.array(keypoint_sequence)
        
        if len(keypoint_sequence) < target_frames:
            # Pad with zeros
            padding = np.zeros((target_frames - len(keypoint_sequence), 
                               keypoint_sequence.shape[1]))
            keypoint_sequence = np.vstack([keypoint_sequence, padding])
        elif len(keypoint_sequence) > target_frames:
            keypoint_sequence = keypoint_sequence[:target_frames]
        
        return keypoint_sequence
    
    def process_dataset(self, json_file='top_100_classes.json', 
                       sequence_length=30, videos_per_class=None):
        """
        Process entire dataset from JSON file (works with local videos only)
        
        Args:
            json_file: JSON file with class data
            sequence_length: Number of frames per sequence
            videos_per_class: Max videos per class (None = all)
        """
        # Load JSON
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        print(f"Processing {len(data)} classes...")
        print(f"Looking for videos in: {self.video_dir}")
        
        # Statistics
        total_processed = 0
        total_failed = 0
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
        
        for class_idx, class_data in enumerate(data):
            gloss = class_data['gloss']
            instances = class_data['instances']
            
            # Limit videos per class if specified
            if videos_per_class:
                instances = instances[:videos_per_class]
            
            print(f"\n[{class_idx+1}/{len(data)}] Processing class: {gloss} "
                  f"({len(instances)} videos)")
            
            # Create class directory
            class_dir = self.output_dir / gloss
            class_dir.mkdir(exist_ok=True)
            
            class_processed = 0
            
            for video_idx, instance in enumerate(instances):
                video_id = str(instance['video_id'])
                frame_start = instance.get('frame_start', None)
                frame_end = instance.get('frame_end', None)
                
                # Handle frame_start = -1 or invalid values
                if frame_start == -1:
                    frame_start = None
                if frame_end == -1:
                    frame_end = None
                
                # Find video file with any extension
                video_path = None
                for ext in video_extensions:
                    test_path = self.video_dir / f"{video_id}{ext}"
                    if test_path.exists():
                        video_path = test_path
                        break
                
                if video_path is None:
                    total_failed += 1
                    continue
                
                # Process video
                try:
                    keypoints = self.process_video(
                        video_path, 
                        frame_start, 
                        frame_end,
                        target_frames=sequence_length
                    )
                    
                    if keypoints is not None:
                        # Save keypoints with proper naming
                        output_file = class_dir / f"{video_id}.npy"
                        np.save(output_file, keypoints)
                        total_processed += 1
                        class_processed += 1
                        
                        if class_processed % 10 == 0:
                            print(f"  Processed {class_processed}/{len(instances)} videos")
                    else:
                        total_failed += 1
                except Exception as e:
                    print(f"  Error processing {video_id}: {str(e)}")
                    total_failed += 1
            
            print(f"  Completed: {class_processed} videos for '{gloss}'")
        
        print(f"\n=== Processing Complete ===")
        print(f"Total videos processed: {total_processed}")
        print(f"Total videos failed: {total_failed}")
        print(f"Output directory: {self.output_dir}")
        
        # Save metadata
        metadata = {
            'num_classes': len(data),
            'sequence_length': sequence_length,
            'total_processed': total_processed,
            'total_failed': total_failed,
            'classes': [item['gloss'] for item in data]
        }
        
        with open(self.output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def create_train_val_split(self, train_ratio=0.8):
        """Create train/validation split"""
        print("\nCreating train/validation split...")
        
        train_dir = self.output_dir / 'train'
        val_dir = self.output_dir / 'val'
        train_dir.mkdir(exist_ok=True)
        val_dir.mkdir(exist_ok=True)
        
        classes = [d for d in self.output_dir.iterdir() 
                  if d.is_dir() and d.name not in ['train', 'val']]
        
        for class_dir in classes:
            class_name = class_dir.name
            files = list(class_dir.glob('*.npy'))
            
            np.random.shuffle(files)
            split_idx = int(len(files) * train_ratio)
            
            train_files = files[:split_idx]
            val_files = files[split_idx:]
            
            # Create class subdirectories
            (train_dir / class_name).mkdir(exist_ok=True)
            (val_dir / class_name).mkdir(exist_ok=True)
            
            # Copy files
            for f in train_files:
                os.link(f, train_dir / class_name / f.name)
            
            for f in val_files:
                os.link(f, val_dir / class_name / f.name)
            
            print(f"  {class_name}: {len(train_files)} train, "
                  f"{len(val_files)} val")
        
        print(f"\nTrain/val split complete!")


def main():
    """Main execution"""
    print("=== WLASL Video Preprocessing ===\n")
    
    # Initialize preprocessor
    preprocessor = WASLVideoPreprocessor(
        video_dir='videos',  # Your video directory
        output_dir='processed_data'
    )
    
    # Process dataset
    preprocessor.process_dataset(
        json_file='top_100_classes.json',
        sequence_length=30,  # Same as ActionDetection tutorial
        videos_per_class=None  # Process all videos (or set a limit)
    )
    
    # Create train/val split
    preprocessor.create_train_val_split(train_ratio=0.8)
    
    print("\n=== Next Steps ===")
    print("1. Check 'processed_data' directory for extracted keypoints")
    print("2. Use train/val folders for model training")
    print("3. Follow ActionDetection tutorial to build LSTM model")
    print("4. Train on your 100 classes!")


if __name__ == "__main__":
    main()
