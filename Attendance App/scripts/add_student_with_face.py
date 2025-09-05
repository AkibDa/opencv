import requests
import base64
import os

API_URL = "http://localhost:3000"
STUDENT_NAME = "Sk Akib Ahammed"
ROLL_NO = "18700124016"
EMAIL = "sk.akib.ahammed.cse.2024@tint.edu.in"
PHOTO_PATH = "/Users/skakibahammed/code_playground/opencv/Attendance App/scripts/18700124016.jpeg"  # Update path

# Step 1: Add or check student
student_data = {
    "name": STUDENT_NAME,
    "roll_no": ROLL_NO,
    "email": EMAIL,
    "photo_url": ""
}

response = requests.post(f"{API_URL}/api/students", json=student_data)
if response.status_code != 200:
    print("Failed to add/check student:", response.text)
    exit()

student_id = response.json().get("student_id")
if not student_id:
    print("Student ID not returned by API.")
    exit()

print(f"Student ID: {student_id} (added or already exists)")

# Step 2: Check if image exists
if not os.path.exists(PHOTO_PATH):
    print(f"Error: Image file not found at '{PHOTO_PATH}'")
    exit()

# Step 3: Read and encode face image
with open(PHOTO_PATH, "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode("utf-8")

# Step 4: Enroll/update face embedding
enroll_data = {
    "user_id": student_id,
    "role": "student",
    "image": img_b64
}

response = requests.post(f"{API_URL}/api/enroll/face", json=enroll_data)
if response.status_code != 200:
    print("Failed to enroll face:", response.text)
    exit()

print(response.json())
print("Face enrolled/updated successfully!")
