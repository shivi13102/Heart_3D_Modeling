import nibabel as nib
import numpy as np
from skimage import measure
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Path to mask file
mask_path = r"E:/SEMESTER SUBJECTS/6th SEMESTER/MINOR PROJECT/Dataset/labelsTr/la_003.nii.gz"

# Load mask
mask = nib.load(mask_path).get_fdata()

# Convert to binary
mask = (mask > 0).astype(np.uint8)

print("Mask shape:", mask.shape)

# Generate 3D surface using marching cubes
verts, faces, normals, values = measure.marching_cubes(mask, level=0.5)

# Plot 3D model
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

ax.plot_trisurf(
    verts[:, 0],
    verts[:, 1],
    faces,
    verts[:, 2],
    color='crimson',   # FIX: use color instead of cmap
    edgecolor='none'
)

ax.set_title("3D Reconstructed Heart – CardioTwin Structural Twin", fontsize=14)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

plt.tight_layout()
plt.show()
