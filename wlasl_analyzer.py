import json
import pandas as pd
import os
from pathlib import Path
from collections import defaultdict

os.chdir(os.path.dirname(os.path.abspath(__file__)))

class WASLDataAnalyzer:
    """
    Analyze WLASL dataset and extract top 100 classes
    Works with LOCAL video files (not URLs)
    """
    
    def __init__(self, json_path='WLASL_v0.3.json', video_dir='videos'):
        self.json_path = json_path
        self.video_dir = Path(video_dir)
        self.data = None
        self.class_stats = None
        
    def check_video_availability(self):
        """Check which videos actually exist in your local folder"""
        print(f"Checking videos in: {self.video_dir}")
        
        if not self.video_dir.exists():
            print(f"ERROR: Video directory not found: {self.video_dir}")
            return None
        
        # Get all video files
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
        available_videos = set()
        
        for ext in video_extensions:
            for video_file in self.video_dir.glob(f'*{ext}'):
                # Extract video_id from filename (without extension)
                video_id = video_file.stem
                available_videos.add(video_id)
        
        print(f"Found {len(available_videos)} video files locally")
        return available_videos
        
    def load_data(self):
        """Load the WLASL JSON file"""
        print(f"\nLoading {self.json_path}...")
        with open(self.json_path, 'r') as f:
            self.data = json.load(f)
        print(f"Loaded {len(self.data)} classes from WLASL dataset")
        return self.data
    
    def analyze_classes(self, available_videos=None):
        """
        Analyze each class and count AVAILABLE video instances
        Only counts videos that actually exist in your folder
        """
        class_info = []
        total_videos_in_json = 0
        total_available_videos = 0
        
        for item in self.data:
            gloss = item['gloss']
            instances = item.get('instances', [])
            total_videos_in_json += len(instances)
            
            # Filter instances to only those with available videos
            available_instances = []
            if available_videos:
                for instance in instances:
                    video_id = str(instance.get('video_id', ''))
                    if video_id in available_videos:
                        available_instances.append(instance)
            else:
                # If we can't check, assume all are available
                available_instances = instances
            
            num_available = len(available_instances)
            total_available_videos += num_available
            
            class_info.append({
                'gloss': gloss,
                'num_videos_total': len(instances),
                'num_videos_available': num_available,
                'instances': available_instances,
                'availability_rate': (num_available / len(instances) * 100) if len(instances) > 0 else 0
            })
        
        # Create DataFrame and sort by AVAILABLE videos
        self.class_stats = pd.DataFrame(class_info)
        self.class_stats = self.class_stats.sort_values('num_videos_available', ascending=False)
        
        print("\n=== Dataset Statistics ===")
        print(f"Total classes: {len(self.class_stats)}")
        print(f"Videos in JSON: {total_videos_in_json}")
        print(f"Videos available locally: {total_available_videos}")
        print(f"Availability rate: {total_available_videos/total_videos_in_json*100:.1f}%")
        
        print(f"\nTop 10 classes by AVAILABLE video count:")
        print(self.class_stats.head(10)[['gloss', 'num_videos_available', 'num_videos_total']])
        
        return self.class_stats
    
    def extract_top_n_classes(self, n=100, min_videos=5, output_json='top_100_classes.json'):
        """
        Extract top N classes with most AVAILABLE videos
        
        Args:
            n: Number of classes to extract
            min_videos: Minimum available videos required per class
            output_json: Output filename
        """
        if self.class_stats is None:
            print("ERROR: Run analyze_classes() first!")
            return None, None
        
        # Filter classes with minimum videos
        filtered_classes = self.class_stats[
            self.class_stats['num_videos_available'] >= min_videos
        ]
        
        if len(filtered_classes) < n:
            print(f"\nWARNING: Only {len(filtered_classes)} classes have {min_videos}+ videos")
            print(f"Adjusting to extract {len(filtered_classes)} classes instead of {n}")
            n = len(filtered_classes)
        
        top_classes = filtered_classes.head(n).copy()
        
        print(f"\n=== Extracting Top {n} Classes ===")
        print(f"Classes: {n}")
        print(f"Total available videos: {top_classes['num_videos_available'].sum()}")
        print(f"Avg videos per class: {top_classes['num_videos_available'].mean():.1f}")
        print(f"Min videos in top {n}: {top_classes['num_videos_available'].min()}")
        print(f"Max videos in top {n}: {top_classes['num_videos_available'].max()}")
        
        # Create new JSON structure with ONLY available instances
        top_data = []
        for idx, row in top_classes.iterrows():
            entry = {
                'gloss': row['gloss'],
                'instances': row['instances']  # Already filtered to available only
            }
            top_data.append(entry)
        
        # Save to new JSON file
        with open(output_json, 'w') as f:
            json.dump(top_data, f, indent=2)
        
        print(f"\nSaved top {n} classes to: {output_json}")
        
        # Save detailed class list
        class_list_file = f'top_{n}_class_list.txt'
        with open(class_list_file, 'w') as f:
            f.write(f"{'Rank':<6}{'Class':<25}{'Available':<12}{'Total':<10}{'Rate':<10}\n")
            f.write("-" * 70 + "\n")
            for rank, (idx, row) in enumerate(top_classes.iterrows(), 1):
                f.write(f"{rank:<6}{row['gloss']:<25}{row['num_videos_available']:<12}"
                       f"{row['num_videos_total']:<10}{row['availability_rate']:<10.1f}%\n")
        
        print(f"Saved class list to: {class_list_file}")
        
        return top_data, top_classes
    
    def create_video_mapping(self, top_classes_df, video_dir, output_csv='video_mapping.csv'):
        """
        Create CSV mapping with verified video paths
        """
        video_data = []
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
        
        for idx, row in top_classes_df.iterrows():
            gloss = row['gloss']
            for instance in row['instances']:
                video_id = str(instance.get('video_id', ''))
                
                # Find actual video file
                video_path = None
                for ext in video_extensions:
                    test_path = Path(video_dir) / f"{video_id}{ext}"
                    if test_path.exists():
                        video_path = str(test_path)
                        break
                
                if video_path:
                    video_data.append({
                        'class_name': gloss,
                        'video_id': video_id,
                        'video_path': video_path,
                        'frame_start': instance.get('frame_start', -1),
                        'frame_end': instance.get('frame_end', -1),
                        'fps': instance.get('fps', 25),
                        'split': instance.get('split', 'unknown')
                    })
        
        video_df = pd.DataFrame(video_data)
        video_df.to_csv(output_csv, index=False)
        print(f"\nCreated video mapping CSV: {output_csv}")
        print(f"Total video mappings: {len(video_df)}")
        
        return video_df
    
    def generate_report(self, top_classes_df, n=100):
        """Generate detailed report"""
        report = f"""
=== WLASL Top {n} Classes Extraction Report ===

Total Classes Selected: {len(top_classes_df)}
Total Available Videos: {top_classes_df['num_videos_available'].sum()}
Average Availability Rate: {top_classes_df['availability_rate'].mean():.1f}%

Video Distribution (Available):
- Mean videos per class: {top_classes_df['num_videos_available'].mean():.1f}
- Median videos per class: {top_classes_df['num_videos_available'].median():.1f}
- Std deviation: {top_classes_df['num_videos_available'].std():.1f}
- Min videos: {top_classes_df['num_videos_available'].min()}
- Max videos: {top_classes_df['num_videos_available'].max()}

Top 20 Classes:
"""
        report += f"{'Rank':<6}{'Class':<25}{'Available':<12}{'Total':<10}{'Rate':<10}\n"
        report += "-" * 70 + "\n"
        
        for idx, (_, row) in enumerate(top_classes_df.head(20).iterrows(), 1):
            report += (f"{idx:<6}{row['gloss']:<25}{row['num_videos_available']:<12}"
                      f"{row['num_videos_total']:<10}{row['availability_rate']:<10.1f}%\n")
        
        report += f"\nBottom 20 of Top {n}:\n"
        report += f"{'Rank':<6}{'Class':<25}{'Available':<12}{'Total':<10}{'Rate':<10}\n"
        report += "-" * 70 + "\n"
        
        for idx, (_, row) in enumerate(top_classes_df.tail(20).iterrows(), 1):
            report += (f"{idx:<6}{row['gloss']:<25}{row['num_videos_available']:<12}"
                      f"{row['num_videos_total']:<10}{row['availability_rate']:<10.1f}%\n")
        
        report_file = f'top_{n}_report.txt'
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(report)
        print(f"Report saved to: {report_file}")


def main():
    """Main execution function"""
    
    print("=" * 70)
    print("WLASL Top 100 Classes Extraction")
    print("Working with LOCAL video files only")
    print("=" * 70)
    
    # Initialize analyzer
    analyzer = WASLDataAnalyzer(
        json_path='WLASL_v0.3.json',
        video_dir='videos'  # Your video folder
    )
    
    # Check which videos are actually available
    available_videos = analyzer.check_video_availability()
    
    if available_videos is None:
        print("\nERROR: Cannot find video directory!")
        print("Make sure 'videos' folder exists with your downloaded videos")
        return
    
    # Load and analyze data
    analyzer.load_data()
    analyzer.analyze_classes(available_videos)
    
    # Extract top 100 classes (with at least 5 videos each)
    top_data, top_classes_df = analyzer.extract_top_n_classes(
        n=100, 
        min_videos=5  # Minimum 5 videos per class
    )
    
    if top_data is None:
        print("\nExtraction failed!")
        return
    
    # Create video mapping with actual file paths
    video_df = analyzer.create_video_mapping(
        top_classes_df, 
        analyzer.video_dir
    )
    
    # Generate report
    analyzer.generate_report(top_classes_df)
    
    print("\n" + "=" * 70)
    print("EXTRACTION COMPLETE!")
    print("=" * 70)
    print("\nGenerated Files:")
    print("1. top_100_classes.json - Filtered dataset with only available videos")
    print("2. video_mapping.csv - Video paths for easy processing")
    print("3. top_100_class_list.txt - Summary of selected classes")
    print("4. top_100_report.txt - Detailed statistics")
    print("\nNext Steps:")
    print("1. Review the report to see your selected classes")
    print("2. Use video_mapping.csv for preprocessing")
    print("3. Run the preprocessing script to extract keypoints")
    print("4. Train your model!")


if __name__ == "__main__":
    main()