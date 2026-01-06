import nibabel as nib
import matplotlib.pyplot as plt

image_path = r"E:/SEMESTER SUBJECTS/6th SEMESTER/MINOR PROJECT/Dataset/imagesTr/la_003.nii.gz"

img = nib.load(image_path)
data = img.get_fdata()

print("MRI Shape:", data.shape)

# show middle slice
slice_index = data.shape[2] // 2

plt.imshow(data[:, :, slice_index], cmap='gray')
plt.title("Cardiac MRI - Middle Slice")
plt.axis('off')
plt.show()