# multi_target_in_crowd_insightface.py
import cv2
import os
from insightface.app import FaceAnalysis

# --- Paths ---
targets_folder = "datasets/targets"         
crowd_folder = "datasets/crowd_images"      
output_folder = "datasets/results"          
os.makedirs(output_folder, exist_ok=True)

# --- Initialize RetinaFace ---
app = FaceAnalysis(name="retinaface_mnet")  # lightweight RetinaFace
app.prepare(ctx_id=0, nms=0.4)  # ctx_id=0 for CPU, nms for overlapping faces

# --- Load target faces ---
target_encodings = {}
for filename in os.listdir(targets_folder):
    if filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
        path = os.path.join(targets_folder, filename)
        img = cv2.imread(path)
        faces = app.get(img)
        if len(faces) > 0:
            # Take the first face detected in target image
            target_encodings[filename] = faces[0].embedding
        else:
            print(f"No face detected in {filename}")

# --- Function to match faces ---
def match_face(face_embedding, target_encodings, threshold=0.6):
    from numpy import linalg as LA
    for name, emb in target_encodings.items():
        dist = LA.norm(face_embedding - emb)
        if dist < threshold:
            return name
    return None

# --- Process crowd images ---
for filename in os.listdir(crowd_folder):
    if filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
        img_path = os.path.join(crowd_folder, filename)
        img = cv2.imread(img_path)
        faces = app.get(img)

        for face in faces:
            x1, y1, x2, y2 = face.bbox.astype(int)
            name = match_face(face.embedding, target_encodings)
            label = name if name else "Unknown"
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, img)
        print(f"Processed {filename}, saved to {output_path}")

print("All images processed!")
