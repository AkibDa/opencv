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
            if name in target_encodings:
                target_encodings[name].append(emb)
            else:
                target_encodings[name] = [emb]

# finalize averages
for name in target_encodings:
    target_encodings[name] = np.mean(target_encodings[name], axis=0)

print(f"Loaded {len(target_encodings)} target identities: {list(target_encodings.keys())}")

# =====================
# Helper: get best dynamic threshold
# =====================
def get_best_threshold(face_embeddings, target_encodings, factor_range=(1.0, 1.5, 0.05)):
    """
    For a set of face embeddings, try multiple factors and return the factor
    that maximizes recognized faces.
    """
    best_factor = factor_range[0]
    max_recognized = 0

    for factor in np.arange(*factor_range):
        recognized_count = 0
        for emb in face_embeddings:
            min_dist = min(LA.norm(emb - target_emb) for target_emb in target_encodings.values())
            if min_dist < min_dist * factor:  # if distance is within scaled threshold
                recognized_count += 1
        if recognized_count > max_recognized:
            max_recognized = recognized_count
            best_factor = factor

    return best_factor

# =====================
# Helper: match with dynamic threshold
# =====================
def match_face_dynamic(embedding, target_encodings, factor=1.1):
    best_match = "Unknown"
    best_dist = float("inf")

    for name, target_emb in target_encodings.items():
        dist = LA.norm(embedding - target_emb)
        if dist < best_dist:
            best_dist = dist
            best_match = name

    dynamic_threshold = best_dist * factor
    if best_dist < dynamic_threshold:
        return best_match, best_dist
    else:
        return "Unknown", best_dist

# =====================
# Process Crowd Images
# =====================
total_detected, total_recognized = 0, 0
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

        face_embeddings = [face.normed_embedding for face in faces]
        best_factor = 1.1  # fallback
        if face_embeddings:
            best_factor = get_best_threshold(face_embeddings, target_encodings)

        print(f"📏 Using dynamic factor: {best_factor:.2f}")

        faces_detected, faces_recognized = 0, 0

        for i, face in enumerate(faces):
            name, dist = match_face_dynamic(face.normed_embedding, target_encodings, factor=best_factor)
            faces_detected += 1

            # crop and save face
            x1, y1, x2, y2 = face.bbox.astype(int)
            cropped_face = img[y1:y2, x1:x2]

            out_name = f"{os.path.splitext(filename)[0]}_face{i+1}_{name}.jpg"
            cv2.imwrite(os.path.join(output_folder, out_name), cropped_face)

            if name != "Unknown":
                faces_recognized += 1
                print(f"✅ Recognized {name} (dist={dist:.2f}) → Saved {out_name}")
            else:
                print(f"❌ Unknown (dist={dist:.2f}) → Saved {out_name}")

        print(f"📷 {filename} -> Detected: {faces_detected}, Recognized: {faces_recognized}")
        total_detected += faces_detected
        total_recognized += faces_recognized

end_time = time.time()
elapsed = end_time - start_time

print("\n=== RECOGNITION SUMMARY ===")
print(f"🕒 Time spent: {elapsed:.2f} seconds")
print(f"👀 Total faces detected: {total_detected}")
print(f"🎯 Total faces recognized: {total_recognized}")
print("===================")
