from huggingface_hub import snapshot_download

# Specify your desired folder path
desired_folder = "E:/SEMESTER SUBJECTS/6th SEMESTER/MINOR PROJECT/Dataset"  # Change this to your desired path

# Download the dataset to your specified folder
local_path = snapshot_download(
    repo_id="Angelou0516/msd-cardiac",
    repo_type="dataset",
    local_dir=desired_folder,  # Add this parameter
    local_dir_use_symlinks=False  # Optional: creates actual copies instead of symlinks
)

print("Downloaded dataset path:", local_path)