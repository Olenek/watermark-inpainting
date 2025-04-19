from tqdm import tqdm
from huggingface_hub import hf_hub_download
import zipfile
import os
import shutil

def extract_zip(zip_path, extract_to):
    os.makedirs(extract_to, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)


def extract_base_name(filename):
    # From 'image-watermark-001.png' => 'image.png'
    return filename.split("-")[0] + ".jpg"


def collect_target_filenames(wm_dir):
    filenames = os.listdir(wm_dir)
    base_names = set()
    for f in filenames:
        if f.endswith(".png") or f.endswith(".jpg"):
            base_names.add(extract_base_name(f))
    return base_names


def prepare_targets(dataset: str):
    natural_dir, train_wm_dir, val_wm_dir = (
        f'data/{dataset}/natural',
        f'data/{dataset}/train_images',
        f'data/{dataset}/val_images',
    )

    # Paths to final destination
    train_target_dir = os.path.join(train_wm_dir, "target")
    val_target_dir = os.path.join(val_wm_dir, "target")

    os.makedirs(train_target_dir, exist_ok=True)
    os.makedirs(val_target_dir, exist_ok=True)

    # Get needed target image names
    train_targets = collect_target_filenames(os.path.join(train_wm_dir, 'image'))
    val_targets = collect_target_filenames(os.path.join(val_wm_dir, 'image'))

    moved_train = 0
    moved_val = 0

    for fname in train_targets:
        src = str(os.path.join(natural_dir, fname))
        dst = str(os.path.join(train_target_dir, fname))
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.move(src, dst)
            moved_train += 1

    for fname in val_targets:
        src = str(os.path.join(natural_dir, fname))
        dst = str(os.path.join(val_target_dir, fname))
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.move(src, dst)
            moved_val += 1

    print(f"✅ Copied {moved_train} train targets and {moved_val} val targets.")

datasets = [
    '10kgray',
    '10khigh',
    '10kmid',
    '27kpng',
]
files = [f'{dataset}.zip' for dataset in datasets]

for fname in tqdm(files, desc=f'Downloading {files}'):
    zip_path = hf_hub_download(
        repo_id="vinthony/watermark-removal-logo",
        filename=fname,
        repo_type="dataset",
        cache_dir="data/__cache__"
    )

    extract_zip(zip_path, 'data/')

for dataset in datasets:
    prepare_targets(dataset)