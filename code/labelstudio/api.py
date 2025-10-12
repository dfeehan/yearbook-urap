import requests

API_TOKEN = "6300902175076cfa5a76e10257490f71d689d15e"
TASK_ID = 185777725

response = requests.get(
  f"https://app.humansignal.com/api/tasks/{TASK_ID}/annotations/",
  headers={
    "Authorization": f"Token {API_TOKEN}"
  },
)

print(response.json())