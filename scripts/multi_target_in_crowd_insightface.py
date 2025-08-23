# multi_target_in_crowd_insightface.py
import cv2
import os
from insightface.app import FaceAnalysis

# --- Paths ---
targets_folder = "../datasets/detected_faces"  # Folder with target faces   
crowd_folder = "../datasets/crowd_images"      
output_folder = "../datasets/results"          
os.makedirs(output_folder, exist_ok=True)

# --- Initialize RetinaFace ---
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0, det_size=(1024, 1024))

# --- Load target faces ---
target_encodings = {}
failed_targets = []

for filename in os.listdir(targets_folder):
    if filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
        path = os.path.join(targets_folder, filename)
        img = cv2.imread(path)
        if img is None:
            print(f"⚠️ Could not read image: {filename}")
            failed_targets.append(filename)
            continue

        # Resize for better face detection
        h, w = img.shape[:2]
        max_dim = 1024
        min_dim = 256
        scale = 1.0

        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
        elif min(h, w) < min_dim:
            scale = min_dim / min(h, w)

        if scale != 1.0:
            new_w = int(w * scale)
            new_h = int(h * scale)
            img = cv2.resize(img, (new_w, new_h))

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = app.get(img_rgb)
        if len(faces) > 0:
            target_encodings[filename] = faces[0].embedding
            print(f"✅ Loaded target face: {filename}")
        else:
            print(f"⚠️ No face detected in target {filename}")
            failed_targets.append(filename)

if not target_encodings:
    print("❌ No valid target faces found. Please add clear frontal face images.")
    exit()

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
        if img is None:
            print(f"⚠️ Could not read crowd image: {filename}")
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = app.get(img_rgb)

        recognized_count = 0
        for face in faces:
            x1, y1, x2, y2 = face.bbox.astype(int)
            name = match_face(face.embedding, target_encodings)
            label = name if name else "Unknown"
            if name:
                recognized_count += 1
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, img)
        print(f"Processed {filename}, recognized {recognized_count}/{len(faces)}, saved to {output_path}")

if failed_targets:
    print("\n⚠️ Some target images failed to load faces:")
    for f in failed_targets:
        print(f" - {f}")

print("🎉 All images processed!")
