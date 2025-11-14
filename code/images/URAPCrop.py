import requests
from io import BytesIO
from PIL import Image, ImageDraw # type: ignore
import os
import pandas as pd
from collections import defaultdict
import json
from dotenv import load_dotenv
import base64
import anthropic

# Load environment variables from .env file
load_dotenv()

# Configuration from environment variables
API_TOKEN = os.getenv('LABEL_STUDIO_TOKEN')
TASK_ID = int(os.getenv('TASK_ID', 185777732))
OUTPUT_BASE_DIR = os.getenv('OUTPUT_BASE_DIR', os.getcwd())
BASE_URL = os.getenv('LABEL_STUDIO_BASE_URL', 'https://app.humansignal.com')
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')

# Validate required environment variables
if not API_TOKEN:
    raise ValueError("LABEL_STUDIO_TOKEN environment variable is required. Please check your .env file.")

# Initialize Claude client (optional - only if API key is provided)
claude_client = None
if ANTHROPIC_API_KEY and ANTHROPIC_API_KEY != 'your_anthropic_api_key_here':
    try:
        claude_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        print("Claude 3.5 Sonnet initialized for AI text extraction")
    except Exception as e:
        print(f"⚠️ Claude initialization failed: {e}")
        print("Continuing without AI text extraction...")
else:
    print("ℹ️ No Claude API key provided. Skipping AI text extraction.")

# ------------------------------------------------------------------
# 1. Fetch real yearbook annotation data from Label Studio API
# ------------------------------------------------------------------
def fetch_yearbook_annotations():
    """Fetch annotation data from Label Studio API"""
    response = requests.get(
        f"{BASE_URL}/api/tasks/{TASK_ID}/annotations/",
        headers={"Authorization": f"Token {API_TOKEN}"}
    )
    response.raise_for_status()
    return response.json()

def get_task_image_url():
    """Get the actual yearbook page image URL"""
    response = requests.get(
        f"{BASE_URL}/api/tasks/{TASK_ID}/",
        headers={"Authorization": f"Token {API_TOKEN}"}
    )
    response.raise_for_status()
    task_data = response.json()
    
    # Extract the image URL from the task data
    if 'data' in task_data and 'image' in task_data['data']:
        image_path = task_data['data']['image']
        # If it's already a full URL, return it; otherwise prepend BASE_URL
        if image_path.startswith('http'):
            return image_path
        else:
            return f"{BASE_URL}{image_path}"
    
    return None

def extract_text_with_claude(image_crop, text_type="text"):
    """
    Use Claude 3.5 Sonnet to extract and identify text from image regions.
    
    Args:
        image_crop: PIL Image of the text region
        text_type: "name" or "additional" to provide context to Claude
    
    Returns:
        str: Extracted text or None if extraction fails
    """
    if not claude_client:
        return None
    
    try:
        # Convert PIL image to base64
        buffer = BytesIO()
        image_crop.save(buffer, format='PNG')
        image_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        # Create appropriate prompt based on text type
        if text_type.lower() == "name":
            prompt = """This is a cropped image from a historical yearbook containing a student's name. 
Please extract the name exactly as written. The text might be in various fonts or even handwritten.
Respond with ONLY the name, no additional text or explanation."""
        else:
            prompt = """This is a cropped image from a historical yearbook containing additional information about a student 
(such as activities, hometown, major, etc.). Please extract all the text exactly as written.
The text might be in various fonts or even handwritten.
Respond with ONLY the extracted text, no additional explanation."""
        
        # Send to Claude
        response = claude_client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=300,
            messages=[
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": image_data}}
                ]}
            ]
        )
        
        extracted_text = response.content[0].text.strip()
        return extracted_text if extracted_text else None
        
    except Exception as e:
        print(f"Claude text extraction failed: {e}")
        return None

def organize_by_relations(annotations):
    """
    Use explicit relation annotations to group related elements.
    Returns photo groups with all related elements connected via relations.
    """
    photo_groups = []
    
    for annotation in annotations:
        annotation_id = annotation['id']
        created_by = annotation['created_username']
        results = annotation['result']
        
        # Separate different types of results using their actual IDs
        rectangles = {}  # id -> result
        choices = {}     # id -> result
        texts = {}       # id -> result
        relations = []   # list of relations (relations don't have IDs themselves)
        
        # Process all results
        for result in results:
            result_type = result.get('type')
            
            if result_type == 'relation':
                # Relations don't need IDs - they just reference other elements
                relations.append(result)
            else:
                # Other types need IDs for relation linking
                result_id = result.get('id')
                if not result_id:
                    continue  # Skip if no ID
                
                if result_type == 'rectanglelabels':
                    rectangles[result_id] = result
                elif result_type == 'choices':
                    choices[result_id] = result
                elif result_type == 'textarea':
                    texts[result_id] = result
        
        # Find photo rectangles (student/faculty photos)
        photo_rects = {rid: r for rid, r in rectangles.items() if r.get('from_name') == 'photo_box'}
        text_rects = {rid: r for rid, r in rectangles.items() if r.get('from_name') == 'text_box'}
        
        print(f"  Processing: {len(rectangles)} rectangles, {len(relations)} relations")
        print(f"  Found: {len(photo_rects)} photos, {len(text_rects)} text boxes")
        
        # Build relation graph using the actual from_id and to_id
        relation_graph = defaultdict(list)
        for relation in relations:
            from_id = relation.get('from_id')
            to_id = relation.get('to_id')
            if from_id and to_id:
                relation_graph[from_id].append(to_id)
                relation_graph[to_id].append(from_id)  # bidirectional
        
        print(f"  Built relation graph with {len(relation_graph)} connected elements")
        
        # For each photo, find all connected elements via relations
        for photo_id, photo_rect in photo_rects.items():
            # Use DFS to find all connected components
            visited = set()
            connected_elements = []
            
            def dfs(element_id):
                if element_id in visited:
                    return
                visited.add(element_id)
                connected_elements.append(element_id)
                for connected_id in relation_graph.get(element_id, []):
                    dfs(connected_id)
            
            dfs(photo_id)
            
            # Organize connected elements by type
            photo_group = {
                'annotation_id': annotation_id,
                'created_by': created_by,
                'photo_rect': photo_rect,
                'photo_id': photo_id,
                'choices': {},
                'related_texts': [],
                'text_rects': [],
                'connected_ids': connected_elements
            }
            
            # Collect choices and texts that are connected to this photo
            for elem_id in connected_elements:
                if elem_id in choices:
                    choice = choices[elem_id]
                    category = choice.get('from_name')
                    value = choice.get('value', {}).get('choices', [])
                    if category and value:
                        photo_group['choices'][category] = value[0] if value else None
                
                elif elem_id in text_rects:
                    text_rect = text_rects[elem_id]
                    photo_group['text_rects'].append(text_rect)
                
                elif elem_id in texts:
                    text_content = texts[elem_id]
                    photo_group['related_texts'].append(text_content)
            
            photo_groups.append(photo_group)
    
    return photo_groups

print("Fetching yearbook annotation data...")
annotations = fetch_yearbook_annotations()
image_url = get_task_image_url()
print(f"Found {len(annotations)} annotations")
print(f"Image URL: {image_url}")

print("Organizing annotations using explicit relations...")
photo_groups = organize_by_relations(annotations)
print(f"Found {len(photo_groups)} photo regions with related elements")

# Output directories - using environment variable
CROP_DIR = os.path.join(OUTPUT_BASE_DIR, "yearbook_crops")
os.makedirs(CROP_DIR, exist_ok=True)

# ------------------------------------------------------------------
# 2. Load the yearbook page image
# ------------------------------------------------------------------
print("Loading yearbook page image...")
# Add authorization header for image request
headers = {"Authorization": "Token 6300902175076cfa5a76e10257490f71d689d15e"}
resp = requests.get(image_url, timeout=30, headers=headers)
resp.raise_for_status()
img = Image.open(BytesIO(resp.content)).convert("RGB")
W, H = img.size
print(f"Loaded image size: {W} x {H}")

# ------------------------------------------------------------------
# 3. Crop each photo and save with organized metadata + visual relationships
# ------------------------------------------------------------------
def create_filename(photo_group, index, claude_name=None):
    """Create descriptive filename from photo metadata"""
    choices = photo_group['choices']
    
    # Extract key info (may be unknown for this task)
    class_year = choices.get('class_year', 'unknown')
    gender = choices.get('gender', 'unknown')
    race = choices.get('black', 'unknown')
    quality = choices.get('image_quality', 'unknown')
    
    # Get photo label
    photo_rect = photo_group['photo_rect']
    labels = photo_rect.get('value', {}).get('rectanglelabels', ['unknown'])
    photo_type = labels[0] if labels else 'unknown'
    
    # Use Claude-extracted name if available, otherwise try to get name from related text boxes
    if claude_name and claude_name != 'unknown':
        # Clean the name for filename use
        clean_name = claude_name.replace(' ', '_').replace(',', '').replace('.', '').replace("'", "").replace('"', '')
        # Limit length and remove special characters
        clean_name = ''.join(c for c in clean_name if c.isalnum() or c == '_')[:20]
        name_part = clean_name.lower()
    else:
        name_parts = []
        for text_rect in photo_group['text_rects']:
            text_labels = text_rect.get('value', {}).get('rectanglelabels', [])
            if text_labels and text_labels[0] == 'Name':
                # This is a name text box - we don't have the actual text content in this task
                # but we can indicate it's a name box
                name_parts.append('name')
        name_part = '_'.join(name_parts) if name_parts else 'no_name'
    
    # Create filename - if we have no demographic data, focus on what we do have
    if all(v == 'unknown' for v in [class_year, gender, race, quality]):
        # No demographic data available, use text box info instead
        text_types = []
        for text_rect in photo_group['text_rects']:
            text_labels = text_rect.get('value', {}).get('rectanglelabels', [])
            if text_labels:
                text_types.append(text_labels[0].lower().replace(' ', '_'))
        
        text_info = '_'.join(set(text_types)) if text_types else 'no_text'
        filename = f"photo_{index:03d}_{photo_type.replace(' ', '_')}_{name_part}_{text_info}.png"
    else:
        # Use demographic data if available
        filename = f"photo_{index:03d}_{photo_type.replace(' ', '_')}_{class_year}_{gender}_{race}_{quality}.png"
    
    return filename.lower()

def crop_region(img, region_data):
    """Crop a region from the image based on percentage coordinates"""
    W, H = img.size
    
    x_perc = region_data.get('x', 0)
    y_perc = region_data.get('y', 0)
    w_perc = region_data.get('width', 0)
    h_perc = region_data.get('height', 0)
    
    # Convert percentages to pixel coordinates
    left = int((x_perc / 100.0) * W)
    top = int((y_perc / 100.0) * H)
    right = int(left + (w_perc / 100.0) * W)
    bottom = int(top + (h_perc / 100.0) * H)
    
    # Clamp coordinates
    left = max(0, left)
    top = max(0, top)
    right = min(W, right)
    bottom = min(H, bottom)
    
    return img.crop((left, top, right, bottom))

def create_relationship_visual(img, photo_group, index):
    """Create a side-by-side visual showing photo + name + additional text"""
    from PIL import Image, ImageDraw, ImageFont
    
    # Crop the main photo
    photo_rect = photo_group['photo_rect']
    photo_crop = crop_region(img, photo_rect.get('value', {}))
    
    # Find and crop related text regions
    name_crop = None
    additional_crop = None
    
    for text_rect in photo_group['text_rects']:
        text_labels = text_rect.get('value', {}).get('rectanglelabels', [])
        if text_labels:
            text_crop = crop_region(img, text_rect.get('value', {}))
            if text_labels[0] == 'Name':
                name_crop = text_crop
            elif text_labels[0] == 'Additional Text':
                additional_crop = text_crop
    
    # Create a composite image
    padding = 20
    max_height = max(
        photo_crop.height if photo_crop else 100,
        name_crop.height if name_crop else 100,
        additional_crop.height if additional_crop else 100
    )
    
    # Calculate total width
    total_width = padding
    if photo_crop:
        total_width += photo_crop.width + padding
    if name_crop:
        total_width += name_crop.width + padding
    if additional_crop:
        total_width += additional_crop.width + padding
    
    # Create composite image with white background
    composite = Image.new('RGB', (total_width, max_height + 100), 'white')
    draw = ImageDraw.Draw(composite)
    
    # Try to use a default font, fall back to default if not available
    try:
        font = ImageFont.truetype("Arial.ttf", 16)
        title_font = ImageFont.truetype("Arial.ttf", 20)
    except:
        font = ImageFont.load_default()
        title_font = ImageFont.load_default()
    
    # Add title
    title = f"Student {index:03d} - Relationship View"
    draw.text((10, 10), title, fill='black', font=title_font)
    
    # Paste images side by side
    current_x = padding
    y_offset = 40  # Leave space for title
    
    if photo_crop:
        # Center vertically
        y_pos = y_offset + (max_height - photo_crop.height) // 2
        composite.paste(photo_crop, (current_x, y_pos))
        draw.text((current_x, y_pos + photo_crop.height + 5), "PHOTO", fill='red', font=font)
        current_x += photo_crop.width + padding
    
    if name_crop:
        y_pos = y_offset + (max_height - name_crop.height) // 2
        composite.paste(name_crop, (current_x, y_pos))
        draw.text((current_x, y_pos + name_crop.height + 5), "NAME", fill='blue', font=font)
        current_x += name_crop.width + padding
    
    if additional_crop:
        y_pos = y_offset + (max_height - additional_crop.height) // 2
        composite.paste(additional_crop, (current_x, y_pos))
        draw.text((current_x, y_pos + additional_crop.height + 5), "ADDITIONAL", fill='green', font=font)
    
    # Add connection arrows (simple lines)
    if photo_crop and name_crop:
        # Arrow from photo to name
        start_x = padding + photo_crop.width
        end_x = padding + photo_crop.width + padding
        arrow_y = y_offset + max_height // 2
        draw.line([(start_x, arrow_y), (end_x, arrow_y)], fill='gray', width=3)
        draw.polygon([(end_x-10, arrow_y-5), (end_x, arrow_y), (end_x-10, arrow_y+5)], fill='gray')
    
    if name_crop and additional_crop and photo_crop:
        # Arrow from name to additional
        start_x = padding + photo_crop.width + padding + name_crop.width
        end_x = start_x + padding
        arrow_y = y_offset + max_height // 2
        draw.line([(start_x, arrow_y), (end_x, arrow_y)], fill='gray', width=3)
        draw.polygon([(end_x-10, arrow_y-5), (end_x, arrow_y), (end_x-10, arrow_y+5)], fill='gray')
    
    return composite

# Store metadata for CSV export
all_metadata = []

print("Cropping photo regions with relation-based groupings...")

# Create separate directories for different output types
INDIVIDUAL_DIR = os.path.join(CROP_DIR, "individual_photos")
RELATIONSHIPS_DIR = os.path.join(CROP_DIR, "relationship_views")
os.makedirs(INDIVIDUAL_DIR, exist_ok=True)
os.makedirs(RELATIONSHIPS_DIR, exist_ok=True)

for i, photo_group in enumerate(photo_groups):
    photo_rect = photo_group['photo_rect']
    value = photo_rect.get('value', {})
    
    # Extract coordinates (percentages)
    x_perc = value.get('x', 0)
    y_perc = value.get('y', 0)
    w_perc = value.get('width', 0)
    h_perc = value.get('height', 0)
    rot = value.get('rotation', 0)
    
    # Convert percentages to pixel coordinates
    left   = (x_perc / 100.0) * W
    top    = (y_perc / 100.0) * H
    right  = left + (w_perc / 100.0) * W
    bottom = top  + (h_perc / 100.0) * H

    # Clamp and convert to int
    left_i   = max(0, int(round(left)))
    top_i    = max(0, int(round(top)))
    right_i  = min(W, int(round(right)))
    bottom_i = min(H, int(round(bottom)))

    # 1. Crop the individual photo
    crop = img.crop((left_i, top_i, right_i, bottom_i))

    # Handle rotation if needed
    if rot and rot % 360 != 0:
        crop = crop.rotate(-rot, expand=True)

    # Extract text using Claude AI from text box regions (before creating filename)
    claude_extracted_texts = []
    text_boxes_info = []
    
    for text_rect in photo_group['text_rects']:
        text_labels = text_rect.get('value', {}).get('rectanglelabels', [])
        text_value = text_rect.get('value', {})
        text_type = text_labels[0] if text_labels else 'unknown'
        text_coord = f"({text_value.get('x', 0):.1f}%, {text_value.get('y', 0):.1f}%)"
        
        # Use Claude to extract text from this region
        extracted_text = None
        if claude_client:
            try:
                # Crop the text region from the main image
                text_crop = crop_region(img, text_value)
                extracted_text = extract_text_with_claude(text_crop, text_type)
                
                if extracted_text:
                    claude_extracted_texts.append({
                        'type': text_type,
                        'text': extracted_text,
                        'coordinates': text_coord
                    })
                    print(f"    Claude extracted {text_type}: '{extracted_text}'")
                
            except Exception as e:
                print(f"    Claude extraction failed for {text_type}: {e}")
        
        # Store text box info with extracted text
        if extracted_text:
            text_boxes_info.append(f"{text_type} at {text_coord}: '{extracted_text}'")
        else:
            text_boxes_info.append(f"{text_type} at {text_coord}")

    # Get Claude-extracted name for filename
    claude_name = next((item['text'] for item in claude_extracted_texts if item['type'] == 'Name'), None)

    # Create filename with metadata (now including Claude-extracted name)
    filename = create_filename(photo_group, i, claude_name)
    
    # Save individual photo
    individual_path = os.path.join(INDIVIDUAL_DIR, filename)
    crop.save(individual_path)
    
    # 2. Create relationship visual
    relationship_visual = create_relationship_visual(img, photo_group, i)
    relationship_filename = f"relationship_{i:03d}_student_view.png"
    relationship_path = os.path.join(RELATIONSHIPS_DIR, relationship_filename)
    relationship_visual.save(relationship_path)
    
    # Collect text content from related text annotations (manual entries)
    text_content = []
    for text_annotation in photo_group['related_texts']:
        text_val = text_annotation.get('value', {}).get('text', '')
        if text_val and text_val.strip():
            # Clean up the text (remove brackets if present)
            clean_text = text_val.strip().strip("[]'\"")
            text_content.append(clean_text)
    
    # Combine manual and AI-extracted text
    all_extracted_text = text_content + [item['text'] for item in claude_extracted_texts]
    
    # Store comprehensive metadata
    metadata = {
        'index': i,
        'individual_filename': filename,
        'relationship_filename': relationship_filename,
        'annotation_id': photo_group['annotation_id'],
        'created_by': photo_group['created_by'],
        'photo_id': photo_group['photo_id'],
        'photo_type': photo_group['photo_rect'].get('value', {}).get('rectanglelabels', ['unknown'])[0],
        'class_year': photo_group['choices'].get('class_year', 'unknown'),
        'gender': photo_group['choices'].get('gender', 'unknown'),
        'race': photo_group['choices'].get('black', 'unknown'),
        'image_quality': photo_group['choices'].get('image_quality', 'unknown'),
        'coordinates': f"({x_perc:.1f}%, {y_perc:.1f}%)",
        'size': f"{w_perc:.1f}% × {h_perc:.1f}%",
        'pixel_coords': f"({left_i}, {top_i}, {right_i}, {bottom_i})",
        'connected_elements_count': len(photo_group['connected_ids']),
        'related_text_count': len(text_content),
        'ai_extracted_text_count': len(claude_extracted_texts),
        'related_text_boxes': len(photo_group['text_rects']),
        'text_box_details': '; '.join(text_boxes_info),
        'related_texts': '; '.join(text_content) if text_content else 'none',
        'ai_extracted_texts': '; '.join(all_extracted_text) if all_extracted_text else 'none',
        'claude_name': next((item['text'] for item in claude_extracted_texts if item['type'] == 'Name'), 'unknown'),
        'claude_additional': next((item['text'] for item in claude_extracted_texts if item['type'] == 'Additional Text'), 'unknown'),
        'connected_ids': ', '.join(photo_group['connected_ids'])
    }
    all_metadata.append(metadata)
    
    print(f"Saved {filename}: {metadata['photo_type']} - {metadata['gender']} {metadata['class_year']} ({metadata['race']}, {metadata['image_quality']} quality)")
    print(f"  -> Individual: {individual_path}")
    print(f"  -> Relationship: {relationship_path}")
    print(f"  -> Connected to {metadata['connected_elements_count']} elements: {metadata['text_box_details']}")

print(f"\nCropped {len(photo_groups)} photos:")
print(f"  Individual photos: {INDIVIDUAL_DIR}")
print(f"  Relationship views: {RELATIONSHIPS_DIR}")

# ------------------------------------------------------------------
# 4. Create comprehensive metadata CSV
# ------------------------------------------------------------------
df_metadata = pd.DataFrame(all_metadata)
csv_path = os.path.join(CROP_DIR, "yearbook_photo_metadata.csv")
df_metadata.to_csv(csv_path, index=False)
print(f"Saved metadata to {csv_path}")

# ------------------------------------------------------------------
# 5. Create visual overlay showing related elements
# ------------------------------------------------------------------
print("Creating visual overlay with relation-based groupings...")
overlay = img.copy()
draw = ImageDraw.Draw(overlay)

# Use different colors for each photo group
colors = ['red', 'blue', 'green', 'orange', 'purple', 'yellow', 'cyan', 'magenta']

for i, photo_group in enumerate(photo_groups):
    color = colors[i % len(colors)]
    
    # Draw photo rectangle
    photo_rect = photo_group['photo_rect']
    value = photo_rect.get('value', {})
    
    left = int(round((value.get('x', 0) / 100.0) * W))
    top = int(round((value.get('y', 0) / 100.0) * H))
    right = int(round(left + (value.get('width', 0) / 100.0) * W))
    bottom = int(round(top + (value.get('height', 0) / 100.0) * H))
    
    draw.rectangle([left, top, right, bottom], outline=color, width=4)
    
    # Label with metadata
    choices = photo_group['choices']
    label = f"Photo {i}: {choices.get('gender', '?')} {choices.get('class_year', '?')}"
    draw.text((left + 3, top + 3), label, fill=color)
    
    # Draw related text rectangles in same color but thinner
    for text_rect in photo_group['text_rects']:
        text_value = text_rect.get('value', {})
        
        text_left = int(round((text_value.get('x', 0) / 100.0) * W))
        text_top = int(round((text_value.get('y', 0) / 100.0) * H))
        text_right = int(round(text_left + (text_value.get('width', 0) / 100.0) * W))
        text_bottom = int(round(text_top + (text_value.get('height', 0) / 100.0) * H))
        
        draw.rectangle([text_left, text_top, text_right, text_bottom], outline=color, width=2)

overlay_path = os.path.join(CROP_DIR, "yearbook_overlay_relation_groups.png")
overlay.save(overlay_path)
print(f"Saved overlay image with relation-based groupings -> {overlay_path}")

# ------------------------------------------------------------------
# 6. Print summary statistics
# ------------------------------------------------------------------
print("\n" + "="*60)
print("YEARBOOK PHOTO EXTRACTION SUMMARY (RELATION-BASED)")
print("="*60)
print(f"Total photos extracted: {len(photo_groups)}")
print(f"Individual photos saved to: {INDIVIDUAL_DIR}")
print(f"Relationship views saved to: {RELATIONSHIPS_DIR}")
print(f"Metadata saved to: {csv_path}")
print(f"Overlay image saved to: {overlay_path}")

# Print breakdown by categories
if all_metadata:
    print(f"\nBreakdown by gender:")
    gender_counts = pd.Series([m['gender'] for m in all_metadata]).value_counts()
    for gender, count in gender_counts.items():
        print(f"  {gender}: {count}")
    
    print(f"\nBreakdown by class year:")
    year_counts = pd.Series([m['class_year'] for m in all_metadata]).value_counts()
    for year, count in year_counts.items():
        print(f"  {year}: {count}")
    
    print(f"\nBreakdown by connected elements:")
    connection_counts = pd.Series([m['connected_elements_count'] for m in all_metadata]).value_counts()
    for count, photos in connection_counts.items():
        print(f"  {count} connected elements: {photos} photos")

print(f"\nFirst 5 photos with relation info:")
for i, metadata in enumerate(all_metadata[:5]):
    print(f"  {i+1}. Individual: {metadata['individual_filename']}")
    print(f"     Relationship: {metadata['relationship_filename']}")
    print(f"     Connected to {metadata['connected_elements_count']} elements: {metadata['related_text_count']} texts, {metadata['related_text_boxes']} text boxes")
    print(f"     Text boxes: {metadata['text_box_details']}")
    if metadata['related_texts'] != 'none':
        print(f"     Text content: {metadata['related_texts']}")

print(f"\nOutput Structure:")
print(f"  📁 {CROP_DIR}/")
print(f"    📁 individual_photos/ - Individual student photo crops")
print(f"    📁 relationship_views/ - Side-by-side relationship visuals")
print(f"  📄 {csv_path} - Complete metadata CSV")
print(f"  🖼️ {overlay_path} - Full page overlay with groupings")

print(f"\nRelationship views show: PHOTO → NAME → ADDITIONAL TEXT with connecting arrows!")
print(f"All done! Check the relationship_views folder for explicit visual connections.")