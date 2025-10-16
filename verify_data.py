import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import cv2
from collections import Counter
import os 

os.chdir(os.path.dirname(os.path.abspath(__file__)))

class DataVerifier:
    """
    Verify preprocessed and augmented data quality
    Visualize keypoints and check for issues
    """
    
    def __init__(self, data_dir='processed_data'):
        self.data_dir = Path(data_dir)
    
    def verify_preprocessing(self):
        """
        Verify preprocessed data
        """
        print("=" * 70)
        print("VERIFYING PREPROCESSED DATA")
        print("=" * 70)
        
        # Check directory structure
        if not self.data_dir.exists():
            print(f"❌ ERROR: Directory not found: {self.data_dir}")
            return False
        
        # Check for train/val split
        train_dir = self.data_dir / 'train'
        val_dir = self.data_dir / 'val'
        
        has_split = train_dir.exists() and val_dir.exists()
        
        if has_split:
            print("✅ Found train/val split")
            train_classes = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
            val_classes = sorted([d.name for d in val_dir.iterdir() if d.is_dir()])
        else:
            print("⚠️  No train/val split found, checking class directories")
            train_classes = sorted([d.name for d in self.data_dir.iterdir() 
                                   if d.is_dir() and d.name not in ['train', 'val']])
            val_classes = []
        
        print(f"\n📊 Dataset Statistics:")
        print(f"  Training classes: {len(train_classes)}")
        if val_classes:
            print(f"  Validation classes: {len(val_classes)}")
        
        # Count files per class
        print("\n📁 Sample counts per class:")
        
        total_train = 0
        total_val = 0
        
        # Check training data - COUNT ALL, SHOW FIRST 10
        class_counts = {}
        for class_name in train_classes:  # Count ALL classes
            if has_split:
                class_dir = train_dir / class_name
            else:
                class_dir = self.data_dir / class_name
            
            npy_files = list(class_dir.glob('*.npy'))
            count = len(npy_files)
            class_counts[class_name] = count
            total_train += count
            
            # Only PRINT first 10
            if len(class_counts) <= 10:
                print(f"  {class_name:20s}: {count:4d} samples")
        
        if len(train_classes) > 10:
            print(f"  ... and {len(train_classes) - 10} more classes")
        
        print(f"\n  Total training samples: {total_train}")
        
        # Check validation data - COUNT ALL
        if val_classes:
            for class_name in val_classes:  # Count ALL validation classes
                class_dir = val_dir / class_name
                npy_files = list(class_dir.glob('*.npy'))
                total_val += len(npy_files)
            print(f"  Total validation samples: {total_val}")
        
        # Load and verify a sample file
        print("\n🔍 Verifying sample data quality...")
        
        sample_class = train_classes[0]
        if has_split:
            sample_dir = train_dir / sample_class
        else:
            sample_dir = self.data_dir / sample_class
        
        sample_files = list(sample_dir.glob('*.npy'))
        
        if not sample_files:
            print("❌ ERROR: No .npy files found!")
            return False
        
        # Check multiple samples
        issues = []
        for i, sample_file in enumerate(sample_files[:5]):
            try:
                data = np.load(sample_file)
                
                # Check shape
                if data.ndim != 2:
                    issues.append(f"Wrong dimensions: {data.ndim}D (expected 2D)")
                
                # Check sequence length (should be 30 frames)
                if data.shape[0] != 30:
                    issues.append(f"Wrong sequence length: {data.shape[0]} (expected 30)")
                
                # Check features (should be 1662)
                expected_features = 33*4 + 468*3 + 21*3 + 21*3  # 1662
                if data.shape[1] != expected_features:
                    issues.append(f"Wrong feature count: {data.shape[1]} (expected {expected_features})")
                
                # Check for NaN or Inf
                if np.isnan(data).any():
                    issues.append("Contains NaN values")
                if np.isinf(data).any():
                    issues.append("Contains Inf values")
                
                # Check if all zeros (invalid)
                if np.all(data == 0):
                    issues.append("All zeros (no keypoints detected)")
                
                if i == 0 and not issues:
                    print(f"  ✅ Sample shape: {data.shape}")
                    print(f"  ✅ Data type: {data.dtype}")
                    print(f"  ✅ Value range: [{data.min():.4f}, {data.max():.4f}]")
                    print(f"  ✅ Mean: {data.mean():.4f}, Std: {data.std():.4f}")
                
            except Exception as e:
                issues.append(f"Error loading file: {str(e)}")
        
        if issues:
            print("\n⚠️  Issues found:")
            for issue in set(issues):  # Remove duplicates
                print(f"  - {issue}")
        else:
            print("\n✅ All samples verified successfully!")
        
        # Check metadata if exists
        metadata_file = self.data_dir / 'metadata.json'
        if metadata_file.exists():
            print("\n📄 Metadata found:")
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            print(f"  Classes: {metadata.get('num_classes', 'N/A')}")
            print(f"  Sequence length: {metadata.get('sequence_length', 'N/A')}")
            print(f"  Processed: {metadata.get('total_processed', 'N/A')}")
            print(f"  Failed: {metadata.get('total_failed', 'N/A')}")
        
        print("\n" + "=" * 70)
        return len(issues) == 0
    
    def visualize_keypoints(self, data_dir='processed_data', num_samples=3):
        """
        Visualize keypoints from processed data
        """
        print("\n" + "=" * 70)
        print("VISUALIZING KEYPOINTS")
        print("=" * 70)
        
        data_path = Path(data_dir)
        
        # Find train directory
        if (data_path / 'train').exists():
            train_dir = data_path / 'train'
        else:
            train_dir = data_path
        
        # Get first class
        classes = sorted([d for d in train_dir.iterdir() if d.is_dir()])
        if not classes:
            print("❌ No class directories found!")
            return
        
        sample_class = classes[0]
        print(f"Visualizing class: {sample_class.name}")
        
        # Get sample files
        sample_files = list(sample_class.glob('*.npy'))[:num_samples]
        
        if not sample_files:
            print("❌ No .npy files found!")
            return
        
        fig, axes = plt.subplots(num_samples, 4, figsize=(20, 5*num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        for idx, sample_file in enumerate(sample_files):
            data = np.load(sample_file)
            
            print(f"\nSample {idx+1}: {sample_file.name}")
            print(f"  Shape: {data.shape}")
            
            # Extract different body parts
            pose_start, pose_end = 0, 33*4
            face_start, face_end = pose_end, pose_end + 468*3
            lh_start, lh_end = face_end, face_end + 21*3
            rh_start, rh_end = lh_end, lh_end + 21*3
            
            pose_data = data[:, pose_start:pose_end]
            face_data = data[:, face_start:face_end]
            lh_data = data[:, lh_start:lh_end]
            rh_data = data[:, rh_start:rh_end]
            
            # Plot pose keypoints over time
            axes[idx, 0].plot(pose_data)
            axes[idx, 0].set_title(f'Pose Keypoints (33 landmarks)\n{sample_file.name}')
            axes[idx, 0].set_xlabel('Frame')
            axes[idx, 0].set_ylabel('Value')
            axes[idx, 0].grid(True, alpha=0.3)
            
            # Plot face keypoints over time
            axes[idx, 1].plot(face_data[:, ::10])  # Plot every 10th feature
            axes[idx, 1].set_title(f'Face Keypoints (468 landmarks)')
            axes[idx, 1].set_xlabel('Frame')
            axes[idx, 1].set_ylabel('Value')
            axes[idx, 1].grid(True, alpha=0.3)
            
            # Plot left hand keypoints
            axes[idx, 2].plot(lh_data)
            axes[idx, 2].set_title(f'Left Hand (21 landmarks)')
            axes[idx, 2].set_xlabel('Frame')
            axes[idx, 2].set_ylabel('Value')
            axes[idx, 2].grid(True, alpha=0.3)
            
            # Plot right hand keypoints
            axes[idx, 3].plot(rh_data)
            axes[idx, 3].set_title(f'Right Hand (21 landmarks)')
            axes[idx, 3].set_xlabel('Frame')
            axes[idx, 3].set_ylabel('Value')
            axes[idx, 3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_file = 'preprocessing_verification.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"\n✅ Visualization saved: {output_file}")
        plt.close()
        
        print("=" * 70)
    
    def compare_original_vs_augmented(self, processed_dir='processed_data', 
                                     augmented_dir='augmented_data'):
        """
        Compare original and augmented data
        """
        print("\n" + "=" * 70)
        print("COMPARING ORIGINAL VS AUGMENTED DATA")
        print("=" * 70)
        
        processed_path = Path(processed_dir)
        augmented_path = Path(augmented_dir)
        
        if not augmented_path.exists():
            print(f"⚠️  Augmented directory not found: {augmented_dir}")
            return
        
        # Count files
        def count_samples(base_dir):
            total = 0
            class_counts = {}
            
            train_dir = base_dir / 'train'
            if train_dir.exists():
                for class_dir in train_dir.iterdir():
                    if class_dir.is_dir():
                        count = len(list(class_dir.glob('*.npy')))
                        class_counts[class_dir.name] = count
                        total += count
            else:
                for class_dir in base_dir.iterdir():
                    if class_dir.is_dir() and class_dir.name not in ['train', 'val']:
                        count = len(list(class_dir.glob('*.npy')))
                        class_counts[class_dir.name] = count
                        total += count
            
            return total, class_counts
        
        original_total, original_counts = count_samples(processed_path)
        augmented_total, augmented_counts = count_samples(augmented_path)
        
        print(f"\n📊 Sample Counts:")
        print(f"  Original (processed):  {original_total:5d} samples")
        print(f"  Augmented:             {augmented_total:5d} samples")
        
        if original_total > 0:
            multiplier = augmented_total / original_total
            print(f"  Augmentation factor:   {multiplier:.1f}x")
        
        # Compare specific classes
        print(f"\n📁 Per-class comparison (first 5 classes):")
        print(f"{'Class':<20} {'Original':>10} {'Augmented':>10} {'Factor':>8}")
        print("-" * 50)
        
        for class_name in sorted(original_counts.keys())[:5]:
            orig = original_counts.get(class_name, 0)
            aug = augmented_counts.get(class_name, 0)
            factor = aug / orig if orig > 0 else 0
            print(f"{class_name:<20} {orig:>10} {aug:>10} {factor:>7.1f}x")
        
        # Visualize comparison
        print(f"\n🎨 Creating comparison visualization...")
        
        # Get a sample from same class
        sample_class = list(original_counts.keys())[0]
        
        # Load original
        if (processed_path / 'train' / sample_class).exists():
            orig_dir = processed_path / 'train' / sample_class
        else:
            orig_dir = processed_path / sample_class
        
        orig_files = list(orig_dir.glob('*.npy'))
        
        # Load augmented versions of same video
        if (augmented_path / 'train' / sample_class).exists():
            aug_dir = augmented_path / 'train' / sample_class
        else:
            aug_dir = augmented_path / sample_class
        
        if orig_files and aug_dir.exists():
            # Get base filename (without _aug suffix)
            orig_file = orig_files[0]
            base_name = orig_file.stem
            
            # Find augmented versions
            aug_files = list(aug_dir.glob(f'{base_name}_aug*.npy'))[:3]
            
            if aug_files:
                fig, axes = plt.subplots(1, len(aug_files)+1, figsize=(20, 4))
                
                # Plot original
                orig_data = np.load(orig_file)
                # Plot right hand (most important for ASL)
                rh_start = 33*4 + 468*3 + 21*3
                rh_end = rh_start + 21*3
                
                axes[0].plot(orig_data[:, rh_start:rh_end])
                axes[0].set_title(f'Original\n{orig_file.name}')
                axes[0].set_xlabel('Frame')
                axes[0].set_ylabel('Right Hand Keypoints')
                axes[0].grid(True, alpha=0.3)
                
                # Plot augmented versions
                for idx, aug_file in enumerate(aug_files, 1):
                    aug_data = np.load(aug_file)
                    axes[idx].plot(aug_data[:, rh_start:rh_end])
                    axes[idx].set_title(f'Augmented {idx}\n{aug_file.name}')
                    axes[idx].set_xlabel('Frame')
                    axes[idx].set_ylabel('Right Hand Keypoints')
                    axes[idx].grid(True, alpha=0.3)
                
                plt.suptitle(f'Class: {sample_class} - Original vs Augmented Samples')
                plt.tight_layout()
                
                output_file = 'augmentation_comparison.png'
                plt.savefig(output_file, dpi=150, bbox_inches='tight')
                print(f"  ✅ Saved: {output_file}")
                plt.close()
        
        print("=" * 70)
    
    def check_data_distribution(self, data_dir='augmented_data'):
        """
        Check if data is balanced across classes
        """
        print("\n" + "=" * 70)
        print("DATA DISTRIBUTION ANALYSIS")
        print("=" * 70)
        
        data_path = Path(data_dir)
        
        train_dir = data_path / 'train' if (data_path / 'train').exists() else data_path
        
        class_counts = {}
        for class_dir in train_dir.iterdir():
            if class_dir.is_dir() and class_dir.name not in ['train', 'val']:
                count = len(list(class_dir.glob('*.npy')))
                class_counts[class_dir.name] = count
        
        if not class_counts:
            print("❌ No data found!")
            return
        
        counts = list(class_counts.values())
        
        print(f"\n📊 Distribution Statistics:")
        print(f"  Total classes: {len(class_counts)}")
        print(f"  Total samples: {sum(counts)}")
        print(f"  Min samples: {min(counts)}")
        print(f"  Max samples: {max(counts)}")
        print(f"  Mean samples: {np.mean(counts):.1f}")
        print(f"  Std deviation: {np.std(counts):.1f}")
        
        # Check if balanced
        std_ratio = np.std(counts) / np.mean(counts)
        if std_ratio < 0.2:
            print(f"  ✅ Well balanced (std/mean = {std_ratio:.2f})")
        else:
            print(f"  ⚠️  Imbalanced (std/mean = {std_ratio:.2f})")
        
        # Plot distribution
        plt.figure(figsize=(15, 6))
        
        sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
        classes, counts = zip(*sorted_classes[:30])  # Top 30
        
        plt.bar(range(len(classes)), counts)
        plt.xlabel('Class')
        plt.ylabel('Number of Samples')
        plt.title('Sample Distribution Across Classes (Top 30)')
        plt.xticks(range(len(classes)), classes, rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        output_file = 'data_distribution.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"\n✅ Distribution plot saved: {output_file}")
        plt.close()
        
        print("=" * 70)


def main():
    """
    Run all verification checks
    """
    print("\n" + "=" * 70)
    print("DATA VERIFICATION TOOL")
    print("=" * 70)
    
    verifier = DataVerifier()
    
    # 1. Verify preprocessing
    print("\n1️⃣  CHECKING PREPROCESSED DATA...")
    preprocessing_ok = verifier.verify_preprocessing()
    
    # 2. Visualize keypoints
    print("\n2️⃣  VISUALIZING KEYPOINTS...")
    verifier.visualize_keypoints(num_samples=3)
    
    # 3. Compare augmentation
    print("\n3️⃣  COMPARING ORIGINAL VS AUGMENTED...")
    verifier.compare_original_vs_augmented()
    
    # 4. Check distribution
    print("\n4️⃣  ANALYZING DATA DISTRIBUTION...")
    verifier.check_data_distribution('augmented_data')
    
    print("\n" + "=" * 70)
    print("VERIFICATION COMPLETE!")
    print("=" * 70)
    print("\n📁 Generated files:")
    print("  1. preprocessing_verification.png - Keypoint visualization")
    print("  2. augmentation_comparison.png - Original vs augmented")
    print("  3. data_distribution.png - Class balance check")
    print("\n✅ Review these images to verify data quality!")
    print("=" * 70)


if __name__ == "__main__":
    main()