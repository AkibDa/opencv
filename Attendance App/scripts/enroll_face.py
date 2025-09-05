import requests
import base64

# ---------------- CONFIG ----------------
API_URL = "http://localhost:3000/api/enroll/face"  # Flask endpoint
USER_ID = "68ba788686c7e387e124c7c5"             # _id of the student in MongoDB
ROLE = "student"
PHOTO_PATH = "/Users/skakibahammed/code_playground/opencv/Attendance App/scripts/18700124016.jpeg"          # path to the student's photo
# ----------------------------------------

# Read image and convert to base64
with open(PHOTO_PATH, "rb") as f:
    img_bytes = f.read()
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")

# Prepare JSON payload
payload = {
    "user_id": USER_ID,
    "role": ROLE,
    "image": img_b64
}

# Send POST request
response = requests.post(API_URL, json=payload)

# Print response from Flask
print(response.status_code)
print(response.json())
