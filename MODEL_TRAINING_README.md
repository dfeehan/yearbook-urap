# Model Training Setup

This directory contains tools for training custom yearbook layout detection models.

## What's Here

- **`fetch_annotations_direct.py`** - Export annotations from Label Studio
- **`create_small_subset.py`** - Create 50-image test subset for quick training
- **`download_images.py`** - Download yearbook images from Label Studio
- **`monitor_training.sh`** - Monitor training progress

See [`layout-model-training/README.md`](layout-model-training/README.md) for detailed training instructions.

## Quick Setup

```bash
# 1. Export annotations
python3 fetch_annotations_direct.py

# 2. Create test subset
python3 create_small_subset.py

# 3. Download images
python3 download_images.py

# 4. Train
cd layout-model-training/scripts && ./train_yearbook_small.sh
```

Training the small model takes ~30 minutes on CPU.

## Full Text-Only Training Setup

If you want a larger training/validation set rather than a small smoke-test subset:

```bash
# 1. Export the full annotation set from Label Studio
python3 fetch_annotations_direct.py

# 2. Download all project images
python3 code/labelstudio/import_export_sandbox/download_project_images.py \
	--project-id 158111 \
	--out-dir layout-model-training/data/yearbook/images

# 3. Rebuild the medium split and cleaned text-only JSONs
python3 create_clean_text_training_data.py

# 4. Train on Perlmutter
sbatch layout-model-training/scripts/perlmutter_train_text_only_cleaned.slurm
```

This produces the same kind of larger train/val setup as the local workflow, but regenerates it directly on the target system.
