import nibabel as nib
import matplotlib.pyplot as plt

img_path = r"E:/SEMESTER SUBJECTS/6th SEMESTER/MINOR PROJECT/Dataset/imagesTr/la_003.nii.gz"
mask_path = r"E:/SEMESTER SUBJECTS/6th SEMESTER/MINOR PROJECT/Dataset/labelsTr/la_003.nii.gz"

img = nib.load(img_path).get_fdata()
mask = nib.load(mask_path).get_fdata()

slice_index = img.shape[2] // 2

plt.figure(figsize=(8,8))
plt.imshow(img[:, :, slice_index], cmap='gray')
plt.imshow(mask[:, :, slice_index], cmap='jet', alpha=0.4)
plt.title("MRI with Heart Segmentation Overlay")
plt.axis('off')
plt.show()
