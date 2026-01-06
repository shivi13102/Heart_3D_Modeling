"""
step4_smooth_3d.py
Smoother 3D heart reconstruction with marching cubes optimization
"""
import nibabel as nib
import numpy as np
from skimage import measure
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import gaussian_filter

# Paths
mask_path = r"E:/SEMESTER SUBJECTS/6th SEMESTER/MINOR PROJECT/Dataset/labelsTr/la_003.nii.gz"

# Load mask
mask = nib.load(mask_path).get_fdata()
mask = (mask > 0).astype(np.uint8)

print(f"Mask shape: {mask.shape}")

# Apply Gaussian filter for smoothing
mask_smoothed = gaussian_filter(mask.astype(float), sigma=1.0)

# Marching cubes with smoothing
verts, faces, normals, values = measure.marching_cubes(
    mask_smoothed, 
    level=0.5,
    step_size=1,  # Smaller step = smoother
    allow_degenerate=False
)

print(f"Generated {len(verts)} vertices and {len(faces)} faces")

# Calculate volume
try:
    img = nib.load(mask_path)
    voxel_size = img.header.get_zooms()  # Get actual voxel dimensions
    volume_voxels = np.sum(mask)
    volume_mm3 = volume_voxels * voxel_size[0] * voxel_size[1] * voxel_size[2]
    volume_ml = volume_mm3 / 1000
except:
    # Fallback if header not available
    voxel_size = (1.0, 1.0, 1.0)
    volume_voxels = np.sum(mask)
    volume_mm3 = volume_voxels
    volume_ml = volume_voxels / 1000

# Plot smooth 3D model - SIMPLIFIED VERSION
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Create surface plot without complex shading
mesh = ax.plot_trisurf(
    verts[:, 0],
    verts[:, 1],
    faces,
    verts[:, 2],
    color='crimson',
    edgecolor='none',
    alpha=0.8,
    shade=True,  # Enable basic shading
    linewidth=0
)

# Set labels and title
ax.set_title("Smooth 3D Heart Reconstruction\n(Digital Twin - Structural Model)", fontsize=16, pad=20)
ax.set_xlabel("X (mm)")
ax.set_ylabel("Y (mm)")
ax.set_zlabel("Z (mm)")

# Set equal aspect ratio
max_range = np.array([verts[:, 0].max()-verts[:, 0].min(), 
                      verts[:, 1].max()-verts[:, 1].min(), 
                      verts[:, 2].max()-verts[:, 2].min()]).max() / 2.0
mid_x = (verts[:, 0].max()+verts[:, 0].min()) * 0.5
mid_y = (verts[:, 1].max()+verts[:, 1].min()) * 0.5
mid_z = (verts[:, 2].max()+verts[:, 2].min()) * 0.5
ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

# Set viewing angle
ax.view_init(elev=20, azim=45)

plt.tight_layout()
plt.show()

# Alternative: Even simpler visualization (guaranteed to work)
fig = plt.figure(figsize=(12, 5))

# Plot 1: Wireframe view
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2],
                 color='crimson', alpha=0.8, edgecolor='k', linewidth=0.2)
ax1.set_title("Smooth Heart Mesh")
ax1.set_xlabel("X")
ax1.set_ylabel("Y")
ax1.set_zlabel("Z")

# Plot 2: Solid view
ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2],
                 color='lightcoral', alpha=1.0, edgecolor='none')
ax2.set_title("Solid Heart Model")
ax2.set_xlabel("X")
ax2.set_ylabel("Y")
ax2.set_zlabel("Z")

plt.suptitle("3D Cardiac Digital Twin - Multiple Views", fontsize=14)
plt.tight_layout()
plt.show()

print(f"\n=== Quantitative Results ===")
print(f"Volume in voxels: {volume_voxels}")
print(f"Voxel dimensions (mm): {voxel_size}")
print(f"Volume in mm³: {volume_mm3:.2f}")
print(f"Volume in mL: {volume_ml:.2f}")

# Additional visualization: Compare original vs smoothed
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Original mask slice
slice_idx = mask.shape[2] // 2
axes[0].imshow(mask[:, :, slice_idx], cmap='gray')
axes[0].set_title("Original Binary Mask")
axes[0].axis('off')

# Smoothed mask slice
axes[1].imshow(mask_smoothed[:, :, slice_idx], cmap='gray')
axes[1].set_title("Smoothed Mask (sigma=1.0)")
axes[1].axis('off')

# Difference
diff = mask_smoothed[:, :, slice_idx] - mask[:, :, slice_idx].astype(float)
axes[2].imshow(diff, cmap='coolwarm')
axes[2].set_title("Difference (Smooth - Original)")
axes[2].axis('off')

plt.tight_layout()
plt.show()