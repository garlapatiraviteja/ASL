import numpy as np
import random
import os
from pathlib import Path
import json
from tqdm import tqdm

os.chdir(os.path.dirname(os.path.abspath(__file__)))

class ASLDataAugmentation:
    """
    Augment keypoint sequences to increase training data
    Turns 2,038 videos into 10,000+ training samples
    """
    
    @staticmethod
    def add_noise(keypoints, noise_level=0.01):
        """Add random Gaussian noise to keypoints"""
        noise = np.random.normal(0, noise_level, keypoints.shape)
        return keypoints + noise
    
    @staticmethod
    def time_warp(keypoints, speed_factor=None):
        """Speed up or slow down the sequence (temporal augmentation)"""
        if speed_factor is None:
            speed_factor = random.uniform(0.85, 1.15)
        
        current_length = len(keypoints)
        new_length = int(current_length * speed_factor)
        
        if new_length < 2:
            new_length = 2
        
        # Interpolate to new length
        indices = np.linspace(0, current_length - 1, new_length)
        warped = np.zeros((new_length, keypoints.shape[1]))
        
        for i in range(keypoints.shape[1]):
            warped[:, i] = np.interp(indices, np.arange(current_length), keypoints[:, i])
        
        # Resize back to original length
        final_indices = np.linspace(0, new_length - 1, current_length)
        result = np.zeros_like(keypoints)
        
        for i in range(keypoints.shape[1]):
            result[:, i] = np.interp(final_indices, np.arange(new_length), warped[:, i])
        
        return result
    
    @staticmethod
    def spatial_shift(keypoints, shift_range=0.05):
        """Shift all keypoints in 2D space (simulates camera position change)"""
        shift_x = random.uniform(-shift_range, shift_range)
        shift_y = random.uniform(-shift_range, shift_range)
        
        augmented = keypoints.copy()
        
        # MediaPipe format: [pose(33x4), face(468x3), lh(21x3), rh(21x3)]
        # Total: 132 + 1404 + 63 + 63 = 1662 features
        
        # Shift pose landmarks (x, y coordinates only)
        for i in range(0, 33*4, 4):  # Pose: 33 landmarks x 4 (x,y,z,vis)
            augmented[:, i] += shift_x      # x coordinate
            augmented[:, i + 1] += shift_y  # y coordinate
        
        # Shift face landmarks
        for i in range(33*4, 33*4 + 468*3, 3):  # Face: 468 landmarks x 3
            augmented[:, i] += shift_x
            augmented[:, i + 1] += shift_y
        
        # Shift left hand landmarks
        for i in range(33*4 + 468*3, 33*4 + 468*3 + 21*3, 3):  # LH: 21 x 3
            augmented[:, i] += shift_x
            augmented[:, i + 1] += shift_y
        
        # Shift right hand landmarks
        for i in range(33*4 + 468*3 + 21*3, 33*4 + 468*3 + 21*3 + 21*3, 3):  # RH: 21 x 3
            augmented[:, i] += shift_x
            augmented[:, i + 1] += shift_y
        
        return augmented
    
    @staticmethod
    def rotation_2d(keypoints, angle_range=10):
        """Rotate keypoints around center (simulates head/body tilt)"""
        angle = np.radians(random.uniform(-angle_range, angle_range))
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        
        augmented = keypoints.copy()
        
        # Rotate pose landmarks
        for i in range(0, 33*4, 4):
            x = augmented[:, i]
            y = augmented[:, i + 1]
            augmented[:, i] = x * cos_a - y * sin_a
            augmented[:, i + 1] = x * sin_a + y * cos_a
        
        # Rotate face landmarks
        for i in range(33*4, 33*4 + 468*3, 3):
            x = augmented[:, i]
            y = augmented[:, i + 1]
            augmented[:, i] = x * cos_a - y * sin_a
            augmented[:, i + 1] = x * sin_a + y * cos_a
        
        # Rotate left hand
        for i in range(33*4 + 468*3, 33*4 + 468*3 + 21*3, 3):
            x = augmented[:, i]
            y = augmented[:, i + 1]
            augmented[:, i] = x * cos_a - y * sin_a
            augmented[:, i + 1] = x * sin_a + y * cos_a
        
        # Rotate right hand
        for i in range(33*4 + 468*3 + 21*3, 33*4 + 468*3 + 21*3 + 21*3, 3):
            x = augmented[:, i]
            y = augmented[:, i + 1]
            augmented[:, i] = x * cos_a - y * sin_a
            augmented[:, i + 1] = x * sin_a + y * cos_a
        
        return augmented
    
    @staticmethod
    def scale(keypoints, scale_range=(0.92, 1.08)):
        """Scale keypoints uniformly (simulates distance from camera)"""
        scale_factor = random.uniform(*scale_range)
        return keypoints * scale_factor
    
    @staticmethod
    def dropout_frames(keypoints, dropout_prob=0.1):
        """Randomly interpolate some frames (simulates tracking loss)"""
        augmented = keypoints.copy()
        num_frames = len(keypoints)
        
        for i in range(num_frames):
            if random.random() < dropout_prob:
                # Interpolate from neighbors
                if i > 0 and i < num_frames - 1:
                    augmented[i] = (augmented[i-1] + augmented[i+1]) / 2
                elif i == 0 and num_frames > 1:
                    augmented[i] = augmented[i+1]
                elif i == num_frames - 1 and num_frames > 1:
                    augmented[i] = augmented[i-1]
        
        return augmented
    
    @staticmethod
    def hand_emphasis(keypoints, emphasis_factor=1.2):
        """Emphasize hand movements (important for ASL)"""
        augmented = keypoints.copy()
        
        # Emphasize left hand
        lh_start = 33*4 + 468*3
        lh_end = lh_start + 21*3
        augmented[:, lh_start:lh_end] *= emphasis_factor
        
        # Emphasize right hand
        rh_start = lh_end
        rh_end = rh_start + 21*3
        augmented[:, rh_start:rh_end] *= emphasis_factor
        
        return augmented
    
    @staticmethod
    def gaussian_blur_temporal(keypoints, kernel_size=3):
        """Smooth the temporal sequence (reduces jitter)"""
        if kernel_size < 2:
            return keypoints
            
        augmented = keypoints.copy()
        
        # Apply Gaussian blur along time axis
        for feature_idx in range(keypoints.shape[1]):
            # Simple moving average
            for i in range(len(keypoints)):
                start_idx = max(0, i - kernel_size // 2)
                end_idx = min(len(keypoints), i + kernel_size // 2 + 1)
                augmented[i, feature_idx] = np.mean(keypoints[start_idx:end_idx, feature_idx])
        
        return augmented
    
    @staticmethod
    def combine_augmentations(keypoints, num_augmentations=4, mode='balanced'):
        """
        Apply random combinations of augmentations
        
        Args:
            keypoints: Original keypoint sequence (30 frames x features)
            num_augmentations: Number of augmented versions to create
            mode: 'balanced', 'aggressive', or 'conservative'
        
        Returns:
            List of augmented sequences (includes original)
        """
        augmented_data = []
        
        # Define augmentation strategies
        if mode == 'conservative':
            augmentation_pool = [
                lambda x: ASLDataAugmentation.add_noise(x, 0.005),
                lambda x: ASLDataAugmentation.time_warp(x, 0.95),
                lambda x: ASLDataAugmentation.time_warp(x, 1.05),
                lambda x: ASLDataAugmentation.spatial_shift(x, 0.02),
            ]
        elif mode == 'aggressive':
            augmentation_pool = [
                lambda x: ASLDataAugmentation.add_noise(x, 0.02),
                lambda x: ASLDataAugmentation.time_warp(x, 0.80),
                lambda x: ASLDataAugmentation.time_warp(x, 1.20),
                lambda x: ASLDataAugmentation.spatial_shift(x, 0.08),
                lambda x: ASLDataAugmentation.rotation_2d(x, 20),
                lambda x: ASLDataAugmentation.scale(x, (0.85, 1.15)),
                lambda x: ASLDataAugmentation.dropout_frames(x, 0.15),
            ]
        else:  # balanced (default)
            augmentation_pool = [
                lambda x: ASLDataAugmentation.add_noise(x, 0.01),
                lambda x: ASLDataAugmentation.add_noise(x, 0.015),
                lambda x: ASLDataAugmentation.time_warp(x, 0.90),
                lambda x: ASLDataAugmentation.time_warp(x, 1.10),
                lambda x: ASLDataAugmentation.spatial_shift(x, 0.04),
                lambda x: ASLDataAugmentation.rotation_2d(x, 12),
                lambda x: ASLDataAugmentation.scale(x, (0.94, 1.06)),
                lambda x: ASLDataAugmentation.dropout_frames(x, 0.10),
                lambda x: ASLDataAugmentation.hand_emphasis(x, 1.15),
                lambda x: ASLDataAugmentation.gaussian_blur_temporal(x, 3),
            ]
        
        for _ in range(num_augmentations):
            # Apply 2-3 random augmentations
            num_to_apply = random.randint(2, 3)
            aug_methods = random.sample(augmentation_pool, num_to_apply)
            
            result = keypoints.copy()
            for method in aug_methods:
                result = method(result)
            
            augmented_data.append(result)
        
        return augmented_data


def augment_dataset(input_dir='processed_data', 
                    output_dir='augmented_data', 
                    augmentations_per_video=4,
                    mode='balanced',
                    skip_train_split=False):
    """
    Augment entire processed dataset
    
    Args:
        input_dir: Directory with processed .npy files
        output_dir: Output directory for augmented data
        augmentations_per_video: Number of augmented versions per video
        mode: 'conservative', 'balanced', or 'aggressive'
        skip_train_split: If True, augment train folder only
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True,exist_ok=True)
    
    print("=" * 70)
    print("ASL Data Augmentation")
    print("=" * 70)
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Augmentations per video: {augmentations_per_video}")
    print(f"Mode: {mode}")
    print("=" * 70)
    
    # Determine which directories to process
    if skip_train_split and (input_path / 'train').exists():
        # Only augment training data
        process_dirs = [input_path / 'train']
        output_subdirs = ['train']
    else:
        # Augment all class directories
        process_dirs = [input_path]
        output_subdirs = ['']
    
    total_original = 0
    total_augmented = 0
    
    for proc_dir, out_subdir in zip(process_dirs, output_subdirs):
        # Get all class directories
        class_dirs = [d for d in proc_dir.iterdir() 
                     if d.is_dir() and d.name not in ['train', 'val', 'test']]
        
        print(f"\nProcessing {len(class_dirs)} classes...")
        
        for class_dir in tqdm(class_dirs, desc="Classes"):
            class_name = class_dir.name
            
            # Create output directory
            if out_subdir:
                output_class_dir = output_path / out_subdir / class_name
            else:
                output_class_dir = output_path / class_name
            output_class_dir.mkdir(parents=True, exist_ok=True)
            
            # Get all .npy files
            video_files = list(class_dir.glob('*.npy'))
            
            for video_file in video_files:
                try:
                    # Load original keypoints
                    keypoints = np.load(video_file)
                    total_original += 1
                    
                    # Save original (optional - uncomment if you want originals too)
                    # original_output = output_class_dir / video_file.name
                    # np.save(original_output, keypoints)
                    
                    # Create augmented versions
                    augmented = ASLDataAugmentation.combine_augmentations(
                        keypoints, 
                        num_augmentations=augmentations_per_video,
                        mode=mode
                    )
                    
                    # Save augmented versions
                    for i, aug_keypoints in enumerate(augmented, 1):
                        aug_filename = f"{video_file.stem}_aug{i}.npy"
                        aug_output = output_class_dir / aug_filename
                        np.save(aug_output, aug_keypoints)
                        total_augmented += 1
                        
                except Exception as e:
                    print(f"\nError processing {video_file}: {str(e)}")
                    continue
    
    print("\n" + "=" * 70)
    print("Augmentation Complete!")
    print("=" * 70)
    print(f"Original videos: {total_original}")
    print(f"Augmented videos created: {total_augmented}")
    print(f"Total videos: {total_original + total_augmented}")
    print(f"Multiplication factor: {total_augmented / total_original:.1f}x")
    print(f"\nOutput directory: {output_dir}")


def augment_train_val_separately(base_dir='processed_data',
                                 output_dir='augmented_data',
                                 train_aug=4,
                                 val_aug=0):
    """
    Augment train and validation sets separately
    Typically: augment training data heavily, keep validation original
    
    Args:
        base_dir: Directory with train/val folders
        output_dir: Output directory
        train_aug: Augmentations for training set
        val_aug: Augmentations for validation set (usually 0)
    """
    base_path = Path(base_dir)
    output_path = Path(output_dir)
    
    print("=" * 70)
    print("Augmenting Train/Val Separately")
    print("=" * 70)
    
    # Augment training data
    if train_aug > 0 and (base_path / 'train').exists():
        print(f"\nAugmenting TRAINING data ({train_aug}x)...")
        augment_dataset(
            input_dir=str(base_path / 'train'),
            output_dir=str(output_path / 'train'),
            augmentations_per_video=train_aug,
            mode='balanced'
        )
    
    # Handle validation data
    if (base_path / 'val').exists():
        if val_aug > 0:
            print(f"\nAugmenting VALIDATION data ({val_aug}x)...")
            augment_dataset(
                input_dir=str(base_path / 'val'),
                output_dir=str(output_path / 'val'),
                augmentations_per_video=val_aug,
                mode='conservative'
            )
        else:
            print("\nCopying VALIDATION data (no augmentation)...")
            import shutil
            val_classes = [d for d in (base_path / 'val').iterdir() if d.is_dir()]
            for class_dir in tqdm(val_classes, desc="Copying validation"):
                output_class_dir = output_path / 'val' / class_dir.name
                output_class_dir.mkdir(parents=True, exist_ok=True)
                for npy_file in class_dir.glob('*.npy'):
                    shutil.copy(npy_file, output_class_dir / npy_file.name)
    
    print("\n" + "=" * 70)
    print("Train/Val augmentation complete!")
    print("=" * 70)


def main():
    """
    Main execution
    Choose your augmentation strategy
    """
    print("\n" + "=" * 70)
    print("WLASL Data Augmentation Script")
    print("=" * 70)
    
    # Strategy 1: Augment everything (if you haven't split train/val yet)
    # augment_dataset(
    #     input_dir='processed_data',
    #     output_dir='augmented_data',
    #     augmentations_per_video=4,
    #     mode='balanced'
    # )
    
    # Strategy 2: Augment train/val separately (RECOMMENDED)
    augment_train_val_separately(
        base_dir='processed_data',
        output_dir='augmented_data',
        train_aug=4,  # 4 augmentations for training
        val_aug=0     # 0 augmentations for validation (keep original)
    )
    
    print("\n" + "=" * 70)
    print("Next Steps:")
    print("=" * 70)
    print("1. Check 'augmented_data' folder")
    print("2. Verify file counts match expectations")
    print("3. Start training your model!")
    print("\nExpected results:")
    print("  - Training: ~1,600 original → ~6,400 augmented")
    print("  - Validation: ~400 original (unchanged)")
    print("  - Total: ~6,800 training samples for 100 classes")
    print("=" * 70)


if __name__ == "__main__":
    main()







