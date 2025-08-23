import os
import cv2
import insightface

crowd_folder = "../datasets/crowd_images"
output_folder = "../datasets/detected_faces"
os.makedirs(output_folder, exist_ok=True)

detector = insightface.app.FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
detector.prepare(ctx_id=0, det_size=(640, 640))

for filename in os.listdir(crowd_folder):
  if filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
    image_path = os.path.join(crowd_folder, filename)
    img = cv2.imread(image_path)

    if img is None:
      print(f"[!] Could not read {filename}")
      continue

    print(f"🔍 Processing {filename}...")

    faces = detector.get(img)

    if not faces:
      print(f"❌ No faces detected in {filename}")
      continue

    count = 0
    for i, face in enumerate(faces):
      x1, y1, x2, y2 = face.bbox.astype(int)

      face_crop = img[y1:y2, x1:x2]
      if face_crop.size > 0:
        out_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_face_{i}.jpg")
        cv2.imwrite(out_path, face_crop)
        count += 1

    print(f"✅ {count} faces saved from {filename}")

print("🎉 Done! All crowd images processed.")
