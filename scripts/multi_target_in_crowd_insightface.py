import os
import cv2
import numpy as np
from numpy import linalg as LA
from insightface.app import FaceAnalysis
import time

# =====================
# Config
# =====================
targets_folder = "../datasets/targets"
crowd_folder   = "../datasets/crowd_images"
output_folder  = "../datasets/results"

os.makedirs(output_folder, exist_ok=True)

# =====================
# Initialize InsightFace
# =====================
app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(1024, 1024))

# =====================
# Load Targets (average embeddings if multiple)
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
            emb = faces[0].normed_embedding  # normalized embedding

            # average if already exists
            if name in target_encodings:
                target_encodings[name].append(emb)
            else:
                target_encodings[name] = [emb]

# finalize averages
for name in target_encodings:
    target_encodings[name] = np.mean(target_encodings[name], axis=0)

print(f"Loaded {len(target_encodings)} target identities: {list(target_encodings.keys())}")

# =====================
# Auto threshold based on target distances
# =====================
target_embs = list(target_encodings.values())
if len(target_embs) > 1:
    max_dist_between_targets = max(
        [LA.norm(e1 - e2) for i, e1 in enumerate(target_embs) for j, e2 in enumerate(target_embs) if i != j]
    )
    threshold = min(max_dist_between_targets * 1.05, 1.0)  # small margin
else:
    threshold = 1.0  # default if only 1 target
print(f"Auto threshold set to: {threshold:.2f}")

# =====================
# Helper function
# =====================
def match_face(embedding, target_encodings, threshold):
    best_match = "Unknown"
    best_dist = float("inf")

    for name, target_emb in target_encodings.items():
        dist = LA.norm(embedding - target_emb)

        if dist < best_dist:
            best_dist = dist
            best_match = name

    if best_dist < threshold:
        return best_match, best_dist
    else:
        return "Unknown", best_dist

# =====================
# Process Crowd Images
# =====================
total_faces = 0
total_recognized = 0
start_time = time.time()

for filename in os.listdir(crowd_folder):
    if filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
        path = os.path.join(crowd_folder, filename)
        img = cv2.imread(path)
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        faces = app.get(img_rgb)
        print(f"\n[{filename}] Detected {len(faces)} faces")
        total_faces += len(faces)

        for i, face in enumerate(faces):
            name, dist = match_face(face.normed_embedding, target_encodings, threshold=threshold)

            x1, y1, x2, y2 = face.bbox.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)

            if x2 <= x1 or y2 <= y1:
                print(f"⚠️ Skipping invalid crop for face {i+1}")
                continue

            cropped_face = img[y1:y2, x1:x2]
            out_name = f"{os.path.splitext(filename)[0]}_face{i+1}_{name}.jpg"
            cv2.imwrite(os.path.join(output_folder, out_name), cropped_face)

            if name != "Unknown":
                total_recognized += 1
                print(f"✅ Recognized {name} (dist={dist:.2f}) → Saved {out_name}")
            else:
                print(f"❌ Unknown (dist={dist:.2f}) → Saved {out_name}")

end_time = time.time()

# =====================
# Summary
# =====================
print("\n=== RECOGNITION SUMMARY ===")
print(f"🕒 Time spent: {end_time - start_time:.2f} seconds")
print(f"👀 Total faces detected: {total_faces}")
print(f"🎯 Total faces recognized: {total_recognized}")
print("============================")
