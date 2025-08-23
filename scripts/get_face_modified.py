import os
import time
import cv2
import numpy as np
from insightface.app import FaceAnalysis

crowd_image_path = "../datasets/crowd_images/IMG_8095.jpg"
output_dir       = "../datasets/detected_faces"

DET_SIZE       = (960, 960)   
MIN_FACE_SIZE  = 60          
UPSCALE        = 1.25         
PADDING        = 0.12         

def pad_and_clip_box(x1, y1, x2, y2, w, h, pad=0.1):
  bw, bh = x2 - x1, y2 - y1
  x1 = int(round(max(0, x1 - pad * bw)))
  y1 = int(round(max(0, y1 - pad * bh)))
  x2 = int(round(min(w, x2 + pad * bw)))
  y2 = int(round(min(h, y2 + pad * bh)))
  return x1, y1, x2, y2

def enhance_face(img_bgr):
  lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
  l, a, b = cv2.split(lab)
  clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
  l2 = clahe.apply(l)
  enh = cv2.cvtColor(cv2.merge((l2, a, b)), cv2.COLOR_LAB2BGR)
  kernel = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
  return cv2.filter2D(enh, -1, kernel)

def main():
  os.makedirs(output_dir, exist_ok=True)

  img = cv2.imread(crowd_image_path)
  if img is None:
    print(f"❌ Could not read image: {crowd_image_path}")
    return

  if UPSCALE and UPSCALE != 1.0:
    new_w = int(img.shape[1] * UPSCALE)
    new_h = int(img.shape[0] * UPSCALE)
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

  h, w = img.shape[:2]

  app = FaceAnalysis(
    name="buffalo_l",                 
    allowed_modules=['detection'],    
    providers=['CPUExecutionProvider']
  )
  app.prepare(ctx_id=0, det_size=DET_SIZE)

  t0 = time.time()
  faces = app.get(img)  
  detect_time = time.time() - t0

  total = len(faces)
  print(f"Found {total} face(s) in the crowd image.")

  saved = 0
  for i, f in enumerate(faces, start=1):
    x1, y1, x2, y2 = f.bbox.astype(int)
    bw, bh = x2 - x1, y2 - y1

    if bw < MIN_FACE_SIZE or bh < MIN_FACE_SIZE:
      print(f"  ⏩ Skipping face {i}: too small ({bw}x{bh})")
      continue

    px1, py1, px2, py2 = pad_and_clip_box(x1, y1, x2, y2, w, h, pad=PADDING)
    crop = img[py1:py2, px1:px2]
    if crop.size == 0:
      continue

    crop = enhance_face(crop)

    out_path = os.path.join(output_dir, f"face_{saved+1}.jpg")
    cv2.imwrite(out_path, crop)
    saved += 1

  print(f"✅ {saved} face(es) cropped and saved in '{output_dir}'")
  print(f"⏱ Detection time: {detect_time:.2f}s  (image size: {w}x{h}, det_size={DET_SIZE[0]}x{DET_SIZE[1]}, upscale={UPSCALE})")

if __name__ == "__main__":
  main()
