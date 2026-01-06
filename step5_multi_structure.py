"""
step5_multi_structure.py
Multi-structure cardiac segmentation with 2D and 3D visualization
"""
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure

# Paths
img_path = r"E:/SEMESTER SUBJECTS/6th SEMESTER/MINOR PROJECT/Dataset/imagesTr/la_003.nii.gz"
mask_path = r"E:/SEMESTER SUBJECTS/6th SEMESTER/MINOR PROJECT/Dataset/labelsTr/la_003.nii.gz"

# Load data
img = nib.load(img_path).get_fdata()
mask = nib.load(mask_path).get_fdata()

print(f"Image shape: {img.shape}")
print(f"Mask shape: {mask.shape}")

# Get unique labels in the mask
unique_labels = np.unique(mask)
print(f"Unique labels in mask: {unique_labels}")

# For MSD Cardiac dataset, typical labels are:
# 0: Background
# 1: Left Atrium
# 2: Left Ventricle
# 3: Right Ventricle
# 4: Myocardium
# But let's map based on what we find
label_names = {}
for label in unique_labels:
    if label == 0:
        label_names[label] = "Background"
    elif label == 1:
        label_names[label] = "Left Atrium (LA)"
    elif label == 2:
        label_names[label] = "Left Ventricle (LV)"
    elif label == 3:
        label_names[label] = "Right Ventricle (RV)"
    elif label == 4:
        label_names[label] = "Myocardium (MYO)"
    else:
        label_names[label] = f"Unknown {label}"

print("\nLabel mapping:")
for label, name in label_names.items():
    print(f"  {label}: {name}")

# Create masks for each structure
structures = []
for label in unique_labels:
    if label != 0:  # Skip background
        mask_binary = (mask == label).astype(np.uint8)
        if np.sum(mask_binary) > 0:  # Only include if structure exists
            structures.append({
                'label': label,
                'name': label_names[label],
                'mask': mask_binary,
                'color': plt.cm.tab10(label % 10),
                'voxel_count': np.sum(mask_binary)
            })

print(f"\nFound {len(structures)} cardiac structures:")

# Get voxel dimensions for volume calculation
try:
    header = nib.load(mask_path).header
    voxel_dims = header.get_zooms()
    print(f"Voxel dimensions (mm): {voxel_dims}")
    voxel_volume = voxel_dims[0] * voxel_dims[1] * voxel_dims[2]
except:
    voxel_dims = (1.0, 1.0, 1.0)
    voxel_volume = 1.0
    print("Using default voxel dimensions (1x1x1 mm)")

# Display structure info
for struct in structures:
    print(f"  - {struct['name']}: {struct['voxel_count']:,} voxels")

# ========== 2D MULTI-STRUCTURE VISUALIZATION ==========
slice_idx = img.shape[2] // 2
print(f"\nDisplaying slice {slice_idx} of {img.shape[2]}")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Plot 1: Original MRI
axes[0, 0].imshow(img[:, :, slice_idx], cmap='gray')
axes[0, 0].set_title("Original MRI")
axes[0, 0].axis('off')

# Plot 2: Segmentation mask
axes[0, 1].imshow(mask[:, :, slice_idx], cmap='tab10', vmin=0, vmax=10)
axes[0, 1].set_title("Segmentation Labels")
axes[0, 1].axis('off')

# Plot 3: All structures overlay
axes[0, 2].imshow(img[:, :, slice_idx], cmap='gray', alpha=0.7)
for struct in structures:
    struct_slice = struct['mask'][:, :, slice_idx]
    if np.sum(struct_slice) > 0:
        axes[0, 2].imshow(struct_slice, cmap=plt.cm.colors.ListedColormap([struct['color']]), 
                         alpha=0.5, vmin=0, vmax=1)
axes[0, 2].set_title("All Structures Overlay")
axes[0, 2].axis('off')

# Individual structure plots
for i, struct in enumerate(structures[:3]):  # Show first 3 structures
    row = 1
    col = i
    axes[row, col].imshow(img[:, :, slice_idx], cmap='gray')
    struct_slice = struct['mask'][:, :, slice_idx]
    if np.sum(struct_slice) > 0:
        axes[row, col].imshow(struct_slice, cmap=plt.cm.colors.ListedColormap([struct['color']]), 
                            alpha=0.7, vmin=0, vmax=1)
    axes[row, col].set_title(f"{struct['name']}")
    axes[row, col].axis('off')

# Hide unused subplots
for i in range(len(structures), 3):
    axes[1, i].axis('off')

plt.suptitle("Multi-Structure Cardiac Segmentation - 2D Views", fontsize=16, y=0.98)
plt.tight_layout()
plt.show()

# ========== 3D MULTI-STRUCTURE VISUALIZATION ==========
fig = plt.figure(figsize=(15, 10))

# Create a 3D plot showing all structures
ax = fig.add_subplot(111, projection='3d')

# Track bounding box for setting limits
all_verts = []

for struct in structures:
    if struct['voxel_count'] > 10:  # Only process if enough voxels
        try:
            # Smooth the mask
            mask_smoothed = gaussian_filter(struct['mask'].astype(float), sigma=0.5)
            
            # Generate mesh
            verts, faces, normals, values = measure.marching_cubes(
                mask_smoothed, 
                level=0.5,
                step_size=2,  # Larger step for faster rendering
                allow_degenerate=False
            )
            
            # Create mesh collection
            mesh = Poly3DCollection(verts[faces])
            mesh.set_facecolor(struct['color'])
            mesh.set_edgecolor('black')
            mesh.set_alpha(0.6)
            mesh.set_linewidth(0.1)
            ax.add_collection3d(mesh)
            
            all_verts.append(verts)
            
            print(f"  {struct['name']}: {len(verts)} vertices")
            
        except Exception as e:
            print(f"  Could not generate 3D mesh for {struct['name']}: {str(e)[:50]}...")

# Set plot limits based on all vertices
if all_verts:
    all_verts_array = np.vstack(all_verts)
    max_range = np.array([all_verts_array[:, 0].max()-all_verts_array[:, 0].min(), 
                          all_verts_array[:, 1].max()-all_verts_array[:, 1].min(), 
                          all_verts_array[:, 2].max()-all_verts_array[:, 2].min()]).max() / 2.0
    mid_x = (all_verts_array[:, 0].max()+all_verts_array[:, 0].min()) * 0.5
    mid_y = (all_verts_array[:, 1].max()+all_verts_array[:, 1].min()) * 0.5
    mid_z = (all_verts_array[:, 2].max()+all_verts_array[:, 2].min()) * 0.5
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

ax.set_xlabel("X (mm)")
ax.set_ylabel("Y (mm)")
ax.set_zlabel("Z (mm)")
ax.set_title("3D Multi-Structure Cardiac Digital Twin", fontsize=14)

# Create legend
from matplotlib.patches import Patch
legend_patches = [Patch(color=struct['color'], label=f"{struct['name']}") 
                  for struct in structures if struct['voxel_count'] > 10]
if legend_patches:
    ax.legend(handles=legend_patches, loc='upper left', bbox_to_anchor=(1.05, 1))

# Set viewing angle
ax.view_init(elev=20, azim=30)

plt.tight_layout()
plt.show()

# ========== STRUCTURE VOLUME CALCULATION ==========
print("\n" + "="*60)
print("STRUCTURE VOLUME ANALYSIS")
print("="*60)

for struct in structures:
    volume_mm3 = struct['voxel_count'] * voxel_volume
    volume_ml = volume_mm3 / 1000
    
    # Estimate typical ranges (approximate)
    typical_ranges = {
        'Left Ventricle (LV)': (50, 150),
        'Right Ventricle (RV)': (50, 150),
        'Myocardium (MYO)': (80, 200),
        'Left Atrium (LA)': (30, 100)
    }
    
    typical_min, typical_max = typical_ranges.get(struct['name'], (0, 0))
    
    # Determine status
    if typical_min == 0 and typical_max == 0:
        status = "No typical range available"
    elif typical_min <= volume_ml <= typical_max:
        status = "Within normal range"
    elif volume_ml < typical_min:
        status = f"Low (below {typical_min} mL)"
    else:
        status = f"High (above {typical_max} mL)"
    
    print(f"\n{struct['name']}:")
    print(f"  Label: {struct['label']}")
    print(f"  Voxels: {struct['voxel_count']:,}")
    print(f"  Volume: {volume_mm3:.1f} mm^3 = {volume_ml:.2f} mL")
    if typical_min > 0:
        print(f"  Typical range: {typical_min}-{typical_max} mL -> {status}")

# ========== VOLUME DISTRIBUTION PLOT ==========
plt.figure(figsize=(10, 6))

struct_names = [s['name'] for s in structures]
volumes_ml = [(s['voxel_count'] * voxel_volume / 1000) for s in structures]
colors = [s['color'] for s in structures]

bars = plt.bar(range(len(structures)), volumes_ml, color=colors, alpha=0.8)
plt.xlabel('Cardiac Structure')
plt.ylabel('Volume (mL)')
plt.title('Volume Distribution of Cardiac Structures')
plt.xticks(range(len(structures)), [name.split('(')[0].strip() for name in struct_names], rotation=45)

# Add value labels on top of bars
for bar, volume in zip(bars, volumes_ml):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
             f'{volume:.1f}', ha='center', va='bottom', fontsize=9)

plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()

# ========== STACKED VIEW THROUGH SLICES ==========
print("\nGenerating slice-by-slice view...")

# Create an animation-like view through slices
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

slices_to_show = [slice_idx-10, slice_idx, slice_idx+10]

for i, sl in enumerate(slices_to_show):
    if 0 <= sl < img.shape[2]:
        axes[i].imshow(img[:, :, sl], cmap='gray', alpha=0.7)
        for struct in structures:
            struct_slice = struct['mask'][:, :, sl]
            if np.sum(struct_slice) > 0:
                axes[i].imshow(struct_slice, cmap=plt.cm.colors.ListedColormap([struct['color']]), 
                             alpha=0.5, vmin=0, vmax=1)
        axes[i].set_title(f"Slice {sl}")
        axes[i].axis('off')
    else:
        axes[i].axis('off')

plt.suptitle("Multi-Structure Segmentation Across Slices", fontsize=14)
plt.tight_layout()
plt.show()

print("\n" + "="*60)
print("ANALYSIS COMPLETE")
print("="*60)
total_volume_ml = sum(volumes_ml)
print(f"Total cardiac volume (excluding background): {total_volume_ml:.2f} mL")
print(f"Number of structures identified: {len(structures)}")
print(f"Voxel dimensions: {voxel_dims} mm")

# ========== ADDITIONAL ANALYSIS ==========
if len(structures) < 4:
    print("\n" + "="*60)
    print("NOTE: Only one structure detected in this dataset.")
    print("This dataset appears to be segmented for Left Atrium only.")
    print("For full cardiac analysis, you may need:")
    print("1. A dataset with full heart segmentation")
    print("2. Or use a multi-class segmentation model")
    print("="*60)
    
    # Show what a multi-structure visualization would look like
    print("\nSimulating multi-structure visualization for demonstration...")
    
    # Create simulated structures for demonstration
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # Original
    axes[0].imshow(img[:, :, slice_idx], cmap='gray')
    axes[0].set_title("Original MRI")
    axes[0].axis('off')
    
    # Simulated LV
    axes[1].imshow(img[:, :, slice_idx], cmap='gray')
    # Create a simulated LV mask (circular region)
    y, x = np.ogrid[:img.shape[0], :img.shape[1]]
    center_y, center_x = img.shape[0]//2, img.shape[1]//2
    radius = 50
    mask_lv = (x - center_x)**2 + (y - center_y)**2 <= radius**2
    axes[1].imshow(mask_lv, cmap='Blues', alpha=0.5)
    axes[1].set_title("Simulated LV (Blue)")
    axes[1].axis('off')
    
    # Simulated RV
    axes[2].imshow(img[:, :, slice_idx], cmap='gray')
    # Create a simulated RV mask
    center_y, center_x = img.shape[0]//2 + 30, img.shape[1]//2 - 40
    radius = 40
    mask_rv = (x - center_x)**2 + (y - center_y)**2 <= radius**2
    axes[2].imshow(mask_rv, cmap='Greens', alpha=0.5)
    axes[2].set_title("Simulated RV (Green)")
    axes[2].axis('off')
    
    # Simulated Myocardium
    axes[3].imshow(img[:, :, slice_idx], cmap='gray')
    # Create a simulated myocardium mask (ring)
    radius_outer = 60
    radius_inner = 45
    mask_outer = (x - img.shape[1]//2)**2 + (y - img.shape[0]//2)**2 <= radius_outer**2
    mask_inner = (x - img.shape[1]//2)**2 + (y - img.shape[0]//2)**2 <= radius_inner**2
    mask_myo = mask_outer & ~mask_inner
    axes[3].imshow(mask_myo, cmap='Oranges', alpha=0.5)
    axes[3].set_title("Simulated Myocardium (Orange)")
    axes[3].axis('off')
    
    plt.suptitle("Simulated Multi-Structure Segmentation (for demonstration)", fontsize=14)
    plt.tight_layout()
    plt.show()