# URAPCrop.py - Environment Variable Setup

## Overview
URAPCrop.py has been updated to use environment variables instead of hardcoded paths and API tokens. This makes it easy for multiple people to use the script without modifying the code.

## Setup Instructions

### 1. Install Dependencies
```bash
pip install python-dotenv
```

### 2. Create Your .env File
Copy the example environment file and customize it:
```bash
cp .env.example .env
```

Then edit `.env` with your actual values:
```bash
# Label Studio Configuration
LABEL_STUDIO_TOKEN=your_actual_token_here
TASK_ID=185777732

# Output Configuration  
OUTPUT_BASE_DIR=/path/to/your/output/directory

# API Configuration
LABEL_STUDIO_BASE_URL=https://app.humansignal.com
```

### 3. Environment Variables Explained

- **LABEL_STUDIO_TOKEN**: Your Label Studio API token (required)
- **TASK_ID**: The Label Studio task ID to process (default: 185777732)
- **OUTPUT_BASE_DIR**: Where to save the cropped images and metadata (default: current directory)
- **LABEL_STUDIO_BASE_URL**: Label Studio server URL (default: https://app.humansignal.com)

### 4. Run the Script
```bash
python URAPCrop.py
```

## Output Structure
The script creates the following files in your `OUTPUT_BASE_DIR`:
```
OUTPUT_BASE_DIR/
├── yearbook_crops/
│   ├── individual_photos/          # Individual student photo crops
│   └── relationship_views/         # Side-by-side relationship visuals
├── yearbook_photo_metadata.csv     # Complete metadata
└── yearbook_overlay_relation_groups.png  # Full page overlay
```

## Security Note
- Never commit your `.env` file to git (it's in .gitignore)
- The `.env` file contains your API token, keep it secure
- Use `.env.example` to show others what variables they need

## Troubleshooting

### "LABEL_STUDIO_TOKEN environment variable is required"
- Make sure you created a `.env` file in the project root
- Verify your `.env` file has `LABEL_STUDIO_TOKEN=your_token_here`

### "Invalid URL 'None': No scheme supplied"
- Check that your `LABEL_STUDIO_BASE_URL` is correct
- Verify the task ID exists and you have access to it

### Permission errors
- Make sure the `OUTPUT_BASE_DIR` path exists and is writable
- Check that you have permission to create directories and files