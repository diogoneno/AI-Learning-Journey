# 🛠️ Day 2: API Calls & Authentication

## 📌 Learning Objectives
- Understand what **REST APIs** are.
- Learn how to make **GET/POST requests** using Python (`requests` library).
- Implement **secure authentication** when calling APIs.

---

## 🚀 API Basics
A **REST API** allows different applications to communicate over HTTP using standard methods like:
- **GET** → Retrieve data
- **POST** → Send data
- **PUT** → Update data
- **DELETE** → Remove data





APIs return data in **JSON format**, for example:
```json
{
    "user": "octocat",
    "bio": "GitHub mascot",
    "repos": 42
}




📝 Code Overview

api_authentication.py
->Makes a GET request to the GitHub API.
->Handles API responses and prints user details.
->Demonstrates how to authenticate using an API key.



💾 Running the Code
1. Install the required library (if not installed):

pip install requests


2. Run the script:

python api_authentication.py
