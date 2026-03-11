# Label Studio Import/Export Sandbox

This folder is a safe place to test Label Studio API import/export workflows.

The `export_first_few_photos_and_annotations.py` script is only a small smoke test for checking API access and download behavior. It is not the main dataset-building path for training.

## Script
- `export_first_few_photos_and_annotations.py`
  - Pulls tasks from a Label Studio project
  - Finds photo annotations (`Student Photo`, `Faculty Photo`, `Group Photo`)
  - Downloads the first few source images
  - Saves extracted annotation metadata to JSON

- `import_first_few_images.py`
  - Reads first few image URLs from a text file
  - Prepares Label Studio task payloads
  - Supports dry-run by default and actual import with `--commit`

- `download_project_images.py`
  - Pages through all tasks in a Label Studio project
  - Resolves Label Studio / S3 image URLs
  - Downloads images into `layout-model-training/data/yearbook/images`
  - Safe to rerun; skips existing files unless `--force` is used

## Environment variables
Create a `.env` file in the repo root (or export vars in shell):

- `LABEL_STUDIO_TOKEN` (required)
- `LABEL_STUDIO_BASE_URL` (optional, default: `https://app.humansignal.com`)
- `LABEL_STUDIO_PROJECT_ID` (required unless passed via CLI)

## Usage
From repo root:

`python code/labelstudio/import_export_sandbox/export_first_few_photos_and_annotations.py --limit 5`

`python code/labelstudio/import_export_sandbox/import_first_few_images.py --limit 5`

`python code/labelstudio/import_export_sandbox/download_project_images.py --project-id 158111 --out-dir layout-model-training/data/yearbook/images`

## Full Dataset Workflow
Use this flow when you want a real training/validation dataset like the one used locally:

1. Export all annotations from Label Studio into COCO-style JSON:

`python fetch_annotations_direct.py`

2. Download all project images to disk:

`python code/labelstudio/import_export_sandbox/download_project_images.py --project-id 158111 --out-dir layout-model-training/data/yearbook/images`

3. Build the medium split plus cleaned text-only train/val JSON files:

`python create_clean_text_training_data.py`

4. Train locally or on Perlmutter using the cleaned text-only launcher:

`sbatch layout-model-training/scripts/perlmutter_train_text_only_cleaned.slurm`

Optional args:
- `--project-id 158111`
- `--out-dir code/labelstudio/import_export_sandbox/output`
- `--input-file layout-model-training/data/yearbook/images_to_download.txt`
- `--commit` (for import script, otherwise dry-run)

## Output
- `output/first_few_photo_annotations.json`
- `output/images/...` (downloaded source images)
