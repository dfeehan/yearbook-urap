import requests
import pandas as pd

#Jeremy's API token. Consider using your own, though I'm not sure it really matters
API_TOKEN = "6300902175076cfa5a76e10257490f71d689d15e"

#
TASK_ID = 185777725

response = requests.get(
  f"https://app.humansignal.com/api/tasks/{TASK_ID}/annotations/",
  headers={
    "Authorization": f"Token {API_TOKEN}"
  },
)

data = response.json()
rows = []
for annotation in data:
    annotation_id = annotation['id']
    created_by = annotation['created_username']
    created_at = annotation['created_at']
    
    for result in annotation['result']:
        rows.append({
            'annotation_id': annotation_id,
            'created_by': created_by,
            'created_at': created_at,
            'result_type': result.get('type'),
            'from_name': result.get('from_name'),
            'x': result.get('value', {}).get('x'),
            'y': result.get('value', {}).get('y'),
            'width': result.get('value', {}).get('width'),
            'height': result.get('value', {}).get('height'),
            'choices': str(result.get('value', {}).get('choices')),
            'labels': str(result.get('value', {}).get('rectanglelabels')),
        })

df = pd.DataFrame(rows)
print(df['result_type'].iloc[280:290])
#print(df.iloc[290])

