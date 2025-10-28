import requests

# Example API endpoint (public API)
url = "https://api.github.com/users/octocat"

# Sending a GET request
response = requests.get(url)

# Check response status
if response.status_code == 200:
    data = response.json()
    print("User:", data["login"])
    print("Bio:", data["bio"])
else:
    print("Error:", response.status_code)

# Example of using an API key for authentication (Replace with a real API key)
api_key = "your_api_key_here"
auth_url = "https://api.example.com/data"
headers = {"Authorization": f"Bearer {api_key}"}

auth_response = requests.get(auth_url, headers=headers)

if auth_response.status_code == 200:
    print(auth_response.json())  # Print API response
else:
    print("Failed:", auth_response.status_code)
