import os
import zipfile

from PIL import Image
from huggingface_hub import hf_hub_download
from tqdm import tqdm


def extract_zip(zip_path, extract_to):
    os.makedirs(extract_to, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)


def extract_coco_val_name(filename):
    # From 'COCO_val2014_000000290451-Gant_Logo-159.png' => 'COCO_val2014_000000290451.jpg'
    return filename.split("-")[0] + ".jpg"


def collect_target_filenames(image_dir):
    filenames = os.listdir(image_dir)
    target_names = list()
    for f in filenames:
        target_names.append(extract_coco_val_name(f))
    return filenames, target_names


def prepare_targets(dataset: str):
    natural_dir, train_dir, val_dir = (
        f'data/{dataset}/natural',
        f'data/{dataset}/train_images',
        f'data/{dataset}/val_images',
    )

    # Paths to final destination
    train_target_dir = os.path.join(train_dir, "target")
    val_target_dir = os.path.join(val_dir, "target")

    os.makedirs(train_target_dir, exist_ok=True)
    os.makedirs(val_target_dir, exist_ok=True)

    # Get needed target image names
    train_wm_names, train_target_names = collect_target_filenames(os.path.join(train_dir, 'image'))
    val_wm_names, val_target_names = collect_target_filenames(os.path.join(val_dir, 'image'))

    moved_train = 0
    moved_val = 0

    # noinspection DuplicatedCode
    for wm_name, target_name in tqdm(zip(train_wm_names, train_target_names), total=len(train_wm_names)):
        src = str(os.path.join(natural_dir, target_name))
        dst = str(os.path.join(train_target_dir, wm_name))
        if os.path.exists(src) and not os.path.exists(dst):
            Image.open(src).save(dst)
            moved_train += 1

    # noinspection DuplicatedCode
    for wm_name, target_name in tqdm(zip(val_wm_names, val_target_names), total=len(val_wm_names)):
        src = str(os.path.join(natural_dir, target_name))
        dst = str(os.path.join(val_target_dir, wm_name))
        if os.path.exists(src) and not os.path.exists(dst):
            Image.open(src).save(dst)
            moved_val += 1

    print(f"✅ Copied {moved_train} train targets and {moved_val} val targets.")


datasets = [
    # '10kgray',
    # '10khigh',
    # '10kmid',
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
