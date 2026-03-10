#!/usr/bin/env python3
"""
Check if Label Studio relations are causing duplicate annotations
"""

import json

with open('data/yearbook/annotations.json') as f:
    data = json.load(f)

print('=== Checking for Label Studio relation data ===\n')

# Check top-level keys
print('Top-level keys:', list(data.keys()))

# Check for relations
if 'relations' in data:
    print(f'\n✓ Found "relations" key with {len(data["relations"])} relations')
else:
    print('\n✗ No "relations" key found')

# Look at sample annotation
print('\n=== Sample annotation ===')
sample = data['annotations'][0]
for key, value in sample.items():
    if key == 'bbox':
        print(f'  {key}: {value}')
    elif isinstance(value, (int, float, str)):
        print(f'  {key}: {value}')
    else:
        print(f'  {key}: {type(value)}')

# Check the original Label Studio export file if it exists
import os
ls_files = []
for file in os.listdir('data/yearbook'):
    if 'label' in file.lower() and file.endswith('.json'):
        ls_files.append(file)

if ls_files:
    print(f'\n=== Found Label Studio export files ===')
    for f in ls_files:
        print(f'  {f}')
        # Check first file
        with open(f'data/yearbook/{f}') as lsf:
            ls_data = json.load(lsf)
            if isinstance(ls_data, list) and len(ls_data) > 0:
                print(f'    Format: List of {len(ls_data)} tasks')
                if 'annotations' in ls_data[0]:
                    print(f'    Has annotations field')
                    if len(ls_data[0]['annotations']) > 0:
                        result = ls_data[0]['annotations'][0].get('result', [])
                        print(f'    First result has {len(result)} regions')
                        # Check for relations
                        relations = [r for r in result if r.get('type') == 'relation']
                        if relations:
                            print(f'    ⚠️  Found {len(relations)} relation objects in first task!')
