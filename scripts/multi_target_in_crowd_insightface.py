import os
import cv2
import time
import numpy as np
from numpy import linalg as LA
from insightface.app import FaceAnalysis
from cv2.dnn_superres import DnnSuperResImpl_create

# =====================
# Config
# =====================
targets_folder = "../datasets/targets"
crowd_folder   = "../datasets/crowd_images"
output_folder  = "../datasets/results"
threshold      = 1.0  # tweak this (0.8–1.0)

os.makedirs(output_folder, exist_ok=True)

# =====================
# InsightFace setup
# =====================
app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(1024, 1024))

# =====================
# Super-resolution setup
# =====================
def load_superres(model_path="EDSR_x3.pb"):
    sr = DnnSuperResImpl_create()
    sr.readModel(model_path)
    sr.setModel("edsr", 3)  # 3x upscale
    return sr

sr = load_superres()

def upscale_face(face_img):
    try:
        return sr.upsample(face_img)
    except:
        return face_img

# =====================
# Image enhancement
# =====================
def enhance_image(img):
    alpha, beta = 1.3, 20
    enhanced = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(enhanced, -1, kernel)

# =====================
# Load Targets
# =====================
target_encodings = {}
for filename in os.listdir(targets_folder):
    if filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
        path = os.path.join(targets_folder, filename)
        name = os.path.splitext(filename)[0]
        img = cv2.imread(path)
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = app.get(img_rgb)
        if len(faces) > 0:
            emb = faces[0].normed_embedding
            target_encodings.setdefault(name, []).append(emb)

# Average embeddings if multiple images per target
for name in target_encodings:
    target_encodings[name] = np.mean(target_encodings[name], axis=0)

print(f"Loaded {len(target_encodings)} target identities: {list(target_encodings.keys())}")

# =====================
# Helper function
# =====================
def match_face(embedding, target_encodings, threshold=0.9):
    best_match = "Unknown"
    best_dist = float("inf")
    for name, target_emb in target_encodings.items():
        dist = LA.norm(embedding - target_emb)
        print(f"[DEBUG] Distance to {name}: {dist:.2f}")
        if dist < best_dist:
            best_dist = dist
            best_match = name
    return (best_match if best_dist < threshold else "Unknown", best_dist)

# =====================
# Process Crowd Images
# =====================
recognized_faces_list = []
total_detected, total_recognized = 0, 0
start_time = time.time()  # <<< start timer

for filename in os.listdir(crowd_folder):
    if not filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
        continue

    path = os.path.join(crowd_folder, filename)
    img = cv2.imread(path)
    if img is None:
        continue
    img = enhance_image(img)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    faces = app.get(img_rgb)
    print(f"\n[{filename}] Detected {len(faces)} faces")

    for i, face in enumerate(faces):
        name, dist = match_face(face.normed_embedding, target_encodings, threshold=threshold)

        x1, y1, x2, y2 = face.bbox.astype(int)
        cropped_face = img[y1:y2, x1:x2]

        # Super-resolution for small faces
        if cropped_face.shape[0] < 80 or cropped_face.shape[1] < 80:
            cropped_face = upscale_face(cropped_face)

        if cropped_face is not None and cropped_face.size > 0:
            out_name = f"{os.path.splitext(filename)[0]}_face{i+1}_{name}.jpg"
            cv2.imwrite(os.path.join(output_folder, out_name), cropped_face)
            recognized_faces_list.append(cropped_face)
            print(f" → Saved {out_name} (match: {name}, dist={dist:.2f})")

        total_detected += 1
        if name != "Unknown":
            total_recognized += 1

        # Draw label on crowd image
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, name, (x1, max(y1-10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Save annotated crowd image
    annotated_path = os.path.join(output_folder, f"annotated_{filename}")
    cv2.imwrite(annotated_path, img)

end_time = time.time()  # <<< end timer
elapsed = end_time - start_time

print("\n=== RECOGNITION SUMMARY ===")
print(f"🕒 Time spent: {elapsed:.2f} seconds")
print(f"👀 Total faces detected: {total_detected}")
print(f"🎯 Total faces recognized: {total_recognized}")
