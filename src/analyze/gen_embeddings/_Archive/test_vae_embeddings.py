#!/usr/bin/env python3
"""
VAE Embedding Generation Test Script
Created: August 31, 2025
Purpose: Demonstrate embedding generation workflow from SAM2 training data

This script shows how to:
1. Load training images from Build05 output
2. Generate embeddings (placeholder approach until VAE models are available)
3. Create UMAP visualization
4. Save results for analysis
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from skimage import io
from skimage.feature import local_binary_pattern
from skimage.measure import regionprops, label
from skimage.filters import gaussian
import warnings
warnings.filterwarnings('ignore')

# Try to import UMAP, fallback to PCA if not available
try:
    import umap.umap_ as umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("⚠️  UMAP not available, will use PCA for dimensionality reduction")

class EmbeddingGenerator:
    """Generate morphological embeddings from training data"""
    
    def __init__(self, training_data_path):
        self.training_data_path = Path(training_data_path)
        self.images_path = self.training_data_path / "images"
        self.metadata_path = self.training_data_path / "embryo_metadata_df_train.csv"
        
        # Load metadata if available
        if self.metadata_path.exists():
            self.metadata = pd.read_csv(self.metadata_path)
            print(f"📊 Loaded metadata: {len(self.metadata)} rows")
        else:
            self.metadata = None
            print("⚠️  No metadata file found")
    
    def load_training_images(self):
        """Load all training images and their labels"""
        images = []
        labels = []
        filenames = []
        
        # Walk through category directories
        for category_dir in self.images_path.iterdir():
            if category_dir.is_dir():
                category_name = category_dir.name
                print(f"📁 Processing category: {category_name}")
                
                # Load images from this category
                image_files = list(category_dir.glob("*.jpg")) + list(category_dir.glob("*.png"))
                
                for image_file in image_files:
                    try:
                        image = io.imread(image_file, as_gray=True)
                        images.append(image)
                        labels.append(category_name)
                        filenames.append(image_file.name)
                        print(f"   ✅ Loaded {image_file.name}: {image.shape}")
                    except Exception as e:
                        print(f"   ❌ Failed to load {image_file.name}: {e}")
        
        print(f"📊 Total images loaded: {len(images)}")
        return images, labels, filenames
    
    def extract_morphological_features(self, image):
        """Extract morphological features from a single image"""
        # Ensure image is 2D
        if len(image.shape) > 2:
            image = image[:,:,0] if image.shape[2] > 1 else image.squeeze()
        
        features = {}
        
        # Basic shape properties
        labeled_img = label(image > image.mean())
        props = regionprops(labeled_img, intensity_image=image)
        
        if len(props) > 0:
            # Use the largest region
            largest_region = max(props, key=lambda x: x.area)
            
            features['area'] = largest_region.area
            features['perimeter'] = largest_region.perimeter
            features['eccentricity'] = largest_region.eccentricity
            features['solidity'] = largest_region.solidity
            features['extent'] = largest_region.extent
            features['major_axis_length'] = largest_region.major_axis_length
            features['minor_axis_length'] = largest_region.minor_axis_length
            features['orientation'] = largest_region.orientation
        else:
            # Default values if no regions found
            for key in ['area', 'perimeter', 'eccentricity', 'solidity', 'extent', 
                       'major_axis_length', 'minor_axis_length', 'orientation']:
                features[key] = 0.0
        
        # Texture features using Local Binary Pattern
        try:
            lbp = local_binary_pattern(image, P=8, R=1, method='uniform')
            features['lbp_mean'] = np.mean(lbp)
            features['lbp_std'] = np.std(lbp)
            features['lbp_var'] = np.var(lbp)
        except:
            features['lbp_mean'] = features['lbp_std'] = features['lbp_var'] = 0.0
        
        # Intensity features
        features['mean_intensity'] = np.mean(image)
        features['std_intensity'] = np.std(image)
        features['skewness'] = float(np.mean((image - np.mean(image))**3) / (np.std(image)**3))
        features['kurtosis'] = float(np.mean((image - np.mean(image))**4) / (np.std(image)**4))
        
        # Gradient features
        try:
            blurred = gaussian(image, sigma=1.0)
            grad_y, grad_x = np.gradient(blurred)
            grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            features['gradient_mean'] = np.mean(grad_magnitude)
            features['gradient_std'] = np.std(grad_magnitude)
        except:
            features['gradient_mean'] = features['gradient_std'] = 0.0
        
        return features
    
    def generate_feature_embeddings(self, images, labels, filenames):
        """Generate feature-based embeddings from images"""
        print("🧠 Extracting morphological features...")
        
        feature_list = []
        valid_labels = []
        valid_filenames = []
        
        for i, (image, label, filename) in enumerate(zip(images, labels, filenames)):
            try:
                features = self.extract_morphological_features(image)
                feature_list.append(features)
                valid_labels.append(label)
                valid_filenames.append(filename)
                print(f"   ✅ Extracted features from {filename}")
            except Exception as e:
                print(f"   ❌ Failed to extract features from {filename}: {e}")
        
        # Convert to DataFrame
        features_df = pd.DataFrame(feature_list)
        features_df['label'] = valid_labels
        features_df['filename'] = valid_filenames
        
        print(f"📊 Feature extraction complete: {len(features_df)} samples, {len(features_df.columns)-2} features")
        return features_df
    
    def create_embedding(self, features_df):
        """Create dimensionality reduction embedding from features"""
        embedding_method = "UMAP" if HAS_UMAP else "PCA"
        print(f"🗺️  Creating {embedding_method} embedding...")
        
        # Separate features from labels
        feature_columns = [col for col in features_df.columns if col not in ['label', 'filename']]
        X = features_df[feature_columns].values
        
        # Handle NaN values
        X = np.nan_to_num(X, nan=0.0)
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Create embedding
        if HAS_UMAP:
            reducer = umap.UMAP(n_neighbors=min(5, len(X)-1), 
                               min_dist=0.1, 
                               n_components=2, 
                               random_state=42)
        else:
            reducer = PCA(n_components=2, random_state=42)
        
        embedding = reducer.fit_transform(X_scaled)
        
        # Add embedding coordinates to dataframe
        coord_prefix = "umap" if HAS_UMAP else "pca"
        features_df[f'{coord_prefix}_1'] = embedding[:, 0]
        features_df[f'{coord_prefix}_2'] = embedding[:, 1]
        
        print(f"✅ {embedding_method} embedding created successfully")
        return features_df
    
    def create_visualization(self, features_df, output_path):
        """Create and save UMAP visualization"""
        print("📊 Creating visualization...")
        
        plt.figure(figsize=(12, 8))
        
        # Create subplot layout
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Embedding plot colored by category
        coord_prefix = "umap" if HAS_UMAP else "pca"
        method_name = "UMAP" if HAS_UMAP else "PCA"
        
        scatter = ax1.scatter(features_df[f'{coord_prefix}_1'], features_df[f'{coord_prefix}_2'], 
                             c=pd.Categorical(features_df['label']).codes, 
                             cmap='tab10', alpha=0.7, s=100)
        ax1.set_xlabel(f'{method_name} 1')
        ax1.set_ylabel(f'{method_name} 2')
        ax1.set_title(f'{method_name} Embedding by Category')
        ax1.grid(True, alpha=0.3)
        
        # Add text labels for each point
        for idx, row in features_df.iterrows():
            ax1.annotate(row['filename'].split('_')[2], # Well name like C12, E06
                        (row[f'{coord_prefix}_1'], row[f'{coord_prefix}_2']), 
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, alpha=0.8)
        
        # 2. Feature correlation heatmap
        feature_columns = [col for col in features_df.columns 
                          if col not in ['label', 'filename', f'{coord_prefix}_1', f'{coord_prefix}_2']]
        corr_matrix = features_df[feature_columns].corr()
        
        # Create a simple correlation heatmap using matplotlib
        im = ax2.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
        ax2.set_xticks(range(len(feature_columns)))
        ax2.set_yticks(range(len(feature_columns)))
        ax2.set_xticklabels(feature_columns, rotation=45, ha='right', fontsize=8)
        ax2.set_yticklabels(feature_columns, fontsize=8)
        ax2.set_title('Feature Correlation Matrix')
        plt.colorbar(im, ax=ax2, shrink=0.8)
        
        # 3. Feature importance (variance)
        feature_vars = features_df[feature_columns].var().sort_values(ascending=False)
        ax3.barh(range(len(feature_vars)), feature_vars.values)
        ax3.set_yticks(range(len(feature_vars)))
        ax3.set_yticklabels(feature_vars.index, fontsize=8)
        ax3.set_xlabel('Variance')
        ax3.set_title('Feature Importance (Variance)')
        
        # 4. Sample images
        # Load and display sample images
        ax4.set_title('Sample Training Images')
        ax4.axis('off')
        
        # Create a simple montage of available images
        sample_text = f"Training Data Summary:\n"
        sample_text += f"• Total samples: {len(features_df)}\n"
        sample_text += f"• Categories: {features_df['label'].unique()}\n"
        sample_text += f"• Features extracted: {len(feature_columns)}\n"
        sample_text += f"• Wells: {[f.split('_')[2] for f in features_df['filename']]}\n"
        
        ax4.text(0.1, 0.5, sample_text, transform=ax4.transAxes, 
                fontsize=12, verticalalignment='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        
        plt.tight_layout()
        
        # Save the plot
        output_file = output_path / 'vae_embedding_analysis.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"📊 Visualization saved: {output_file}")
        
        return output_file

def main():
    """Main execution function"""
    print("🧪 VAE Embedding Generation Test")
    print("=" * 50)
    
    # Configuration
    training_data_path = "/net/trapnell/vol1/home/mdcolon/proj/morphseq/morphseq_playground/training_data/sam2_test_20250831_1121"
    output_dir = Path("/net/trapnell/vol1/home/mdcolon/proj/morphseq/safe_test_outputs")
    
    print(f"📁 Training data: {training_data_path}")
    print(f"📁 Output directory: {output_dir}")
    print()
    
    # Initialize embedding generator
    generator = EmbeddingGenerator(training_data_path)
    
    # Load training images
    images, labels, filenames = generator.load_training_images()
    
    if len(images) == 0:
        print("❌ No images found! Check training data path.")
        return
    
    # Generate feature embeddings
    features_df = generator.generate_feature_embeddings(images, labels, filenames)
    
    # Create embedding
    features_df = generator.create_embedding(features_df)
    
    # Create visualization
    viz_file = generator.create_visualization(features_df, output_dir)
    
    # Save feature data
    features_file = output_dir / 'morphological_features.csv'
    features_df.to_csv(features_file, index=False)
    print(f"💾 Features saved: {features_file}")
    
    print()
    print("✅ SUCCESS: Morphological embedding analysis complete!")
    print()
    print("📋 Results Summary:")
    print(f"   • {len(features_df)} embryo snips processed")
    print(f"   • {len(features_df.columns)-4} morphological features extracted")
    print(f"   • UMAP embedding generated")
    print(f"   • Visualization: {viz_file}")
    print(f"   • Features: {features_file}")
    print()
    print("🚀 Next Steps:")
    print("   1. Train actual VAE model on larger dataset")
    print("   2. Replace feature extraction with VAE encoder")
    print("   3. Test with multiple experiments and perturbations")
    print("   4. Validate biological interpretation of embeddings")

if __name__ == "__main__":
    main()