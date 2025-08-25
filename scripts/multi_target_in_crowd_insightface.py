import os
import cv2
import numpy as np
import pandas as pd
import faiss
import logging
from insightface.app import FaceAnalysis
import time

# ---------------- Logging ----------------
logging.basicConfig(
    filename="recognition.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# ---------------- Directories ----------------
targets_folder = "../datasets/college_targets"
crowd_folder = "../datasets/crowd_images"
output_folder = "../datasets/results"
os.makedirs(output_folder, exist_ok=True)

# ---------------- Face Model ----------------
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0, det_size=(640, 640))

# ---------------- Helper: Average multiple target embeddings ----------------
def load_target_embeddings(target_folder):
    target_embeddings = {}
    for filename in os.listdir(target_folder):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            path = os.path.join(target_folder, filename)
            name = os.path.splitext(filename)[0]

            img = cv2.imread(path)
            faces = app.get(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if faces:
                emb = faces[0].embedding / np.linalg.norm(faces[0].embedding)  # normalize
                if name in target_embeddings:
                    target_embeddings[name].append(emb)
                else:
                    target_embeddings[name] = [emb]
                logging.info(f"Loaded target image: {filename}")
    # average embeddings per person
    averaged_embeddings = {}
    for name, embs in target_embeddings.items():
        averaged_embeddings[name] = np.mean(embs, axis=0)
    return averaged_embeddings

target_encodings = load_target_embeddings(targets_folder)
if not target_encodings:
    logging.error("No target faces found!")
    exit()

# ---------------- FAISS index ----------------
names_list = list(target_encodings.keys())
embs_list = np.array(list(target_encodings.values()), dtype="float32")
d = embs_list.shape[1]
index = faiss.IndexFlatL2(d)
index.add(embs_list)

# ---------------- Dynamic threshold helper ----------------
def choose_threshold(distances, min_threshold=0.8, max_threshold=1.5):
    """
    Choose threshold dynamically based on observed distances.
    For very dense/low-quality crowds, increase threshold.
    """
    if len(distances) == 0:
        return min_threshold
    avg_dist = np.mean(distances)
    threshold = min(max(avg_dist * 1.1, min_threshold), max_threshold)
    return threshold

# ---------------- Process Crowd Images ----------------
start_time = time.time()
results_log = []
total_faces = 0
total_recognized = 0

for filename in os.listdir(crowd_folder):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        path = os.path.join(crowd_folder, filename)
        img = cv2.imread(path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = app.get(img_rgb)

        # Gather distances to all targets to choose threshold
        distances = []
        for face in faces:
            embedding = face.embedding / np.linalg.norm(face.embedding)
            D, I = index.search(np.array([embedding], dtype="float32"), 1)
            distances.append(D[0][0])

        threshold = choose_threshold(distances)
        logging.info(f"Processing {filename} with dynamic threshold={threshold:.2f}")

        for i, face in enumerate(faces):
            total_faces += 1
            embedding = face.embedding / np.linalg.norm(face.embedding)
            D, I = index.search(np.array([embedding], dtype="float32"), 1)

            closest_name = names_list[I[0][0]]
            distance = D[0][0]

            if distance < threshold:
                name = closest_name
                total_recognized += 1
                color = (0, 255, 0)
            else:
                name = "Unknown"
                color = (0, 0, 255)

            # Annotate image
            x1, y1, x2, y2 = face.bbox.astype(int)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, f"{name} ({distance:.2f})",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, color, 2)

            # Log CSV
            results_log.append({
                "image": filename,
                "face_id": total_faces,
                "predicted_name": name,
                "distance": distance,
                "threshold_used": threshold
            })

            print(f"[DEBUG] {filename} face {i}: closest={closest_name}, distance={distance:.4f}, threshold={threshold:.2f}")

        out_path = os.path.join(output_folder, f"annotated_{filename}")
        cv2.imwrite(out_path, img)
        logging.info(f"Saved annotated image: {out_path}")

end_time = time.time()

# ---------------- Save CSV ----------------
df = pd.DataFrame(results_log)
csv_path = os.path.join(output_folder, "recognition_results.csv")
df.to_csv(csv_path, index=False)
print(f"📄 CSV log saved to {csv_path}")

# ---------------- SUMMARY ----------------
print("\n=== RECOGNITION SUMMARY ===")
print(f"🕒 Time spent: {end_time - start_time:.2f} seconds")
print(f"👀 Total faces detected: {total_faces}")
print(f"🎯 Total faces recognized: {total_recognized}")
print("============================")
