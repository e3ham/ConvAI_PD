from huggingface_hub import snapshot_download, login

# First, authenticate with Hugging Face
login()  # You'll be prompted for your token

# Download the entire dataset to a local directory
local_dir = "./italian_parkinsons_dataset"
snapshot_download(
    repo_id="birgermoell/Italian_Parkinsons_Voice_and_Speech",
    local_dir=local_dir,
    repo_type="dataset"
)