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

# ---------------- Load target embeddings (multiple images per person) ----------------
def load_target_embeddings(target_folder):
    target_embeddings = {}
    for filename in os.listdir(target_folder):
        if filename.lower().endswith((".jpg", ".jpeg", ".png")):
            path = os.path.join(target_folder, filename)
            name = os.path.splitext(filename)[0]

            img = cv2.imread(path)
            if img is None:
                logging.warning(f"Failed to load target image: {filename}")
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            faces = app.get(img_rgb)
            if faces:
                emb = faces[0].embedding / np.linalg.norm(faces[0].embedding)  # normalize
                if name in target_embeddings:
                    target_embeddings[name].append(emb)
                else:
                    target_embeddings[name] = [emb]
                logging.info(f"Loaded target image: {filename}")

    # Average embeddings per person
    averaged_embeddings = {}
    for name, embs in target_embeddings.items():
        averaged_embeddings[name] = np.mean(embs, axis=0)
    return averaged_embeddings

target_encodings = load_target_embeddings(targets_folder)
if not target_encodings:
    logging.error("No target faces found!")
    exit()

names_list = list(target_encodings.keys())
embs_list = np.array(list(target_encodings.values()), dtype="float32")
d = embs_list.shape[1]

# ---------------- FAISS index ----------------
index = faiss.IndexFlatL2(d)
index.add(embs_list)

# ---------------- Thresholding ----------------
BASE_THRESHOLD = 1.0     # typical for dense crowds
MAX_THRESHOLD = 1.3      # cap for small faces
MIN_THRESHOLD = 0.65     # for high-res/large faces
RATIO_TEST = 0.8         # k-NN ratio threshold

# ---------------- Process Crowd Images ----------------
start_time = time.time()
results_log = []
total_faces = 0
total_recognized = 0

for filename in os.listdir(crowd_folder):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        path = os.path.join(crowd_folder, filename)
        img = cv2.imread(path)
        if img is None:
            logging.warning(f"Failed to load crowd image: {filename}")
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = app.get(img_rgb)

        for i, face in enumerate(faces):
            total_faces += 1
            x1, y1, x2, y2 = face.bbox.astype(int)
            face_size = (x2 - x1) * (y2 - y1)

            # Skip tiny faces
            if face_size < 400:  # ~20x20 pixels
                name = "Unknown"
                distance = np.nan
                threshold = np.nan
                closest_name = "N/A"
                ratio = np.nan
                logging.info(f"Skipped tiny face {i+1} in {filename}")
            else:
                embedding = face.embedding / np.linalg.norm(face.embedding)

                # Search top 3 closest targets
                D, I = index.search(np.array([embedding], dtype="float32"), k=3)

                distance = D[0][0]
                closest_name = names_list[I[0][0]]

                # Adjust threshold based on face size
                threshold = BASE_THRESHOLD
                if face_size < 4000:  # small faces
                    threshold = max(MAX_THRESHOLD, threshold)
                elif face_size > 10000:  # large faces
                    threshold = MIN_THRESHOLD

                # Ratio test: ensure the closest is significantly better than 2nd closest
                if D[0][1] == 0:
                    ratio = 0
                else:
                    ratio = D[0][0] / (D[0][1] + 1e-6)

                if distance < threshold and ratio < RATIO_TEST:
                    name = closest_name
                    total_recognized += 1
                    color = (0, 255, 0)
                else:
                    name = "Unknown"
                    color = (0, 0, 255)

                # Annotate image
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, f"{name} ({distance:.2f})",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, color, 2)

            # Log results
            results_log.append({
                "image": filename,
                "face_id": total_faces,
                "predicted_name": name,
                "distance": distance,
                "threshold_used": threshold
            })

            print(f"[DEBUG] {filename} face {i+1}: closest={closest_name}, distance={distance:.4f}, threshold={threshold:.2f}, ratio={ratio:.2f}")

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
