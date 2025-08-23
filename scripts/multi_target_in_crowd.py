import face_recognition
import cv2
import os
import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import freeze_support

# --- Helper Functions ---

def process_single_target_image(path):
    """Load and encode a single target image. Designed for parallel processing."""
    image = face_recognition.load_image_file(path)
    encodings = face_recognition.face_encodings(image)
    if encodings:
        return encodings[0], os.path.splitext(os.path.basename(path))[0]
    return None, None

def preprocess_image(image):
    """Enhance image for better detection in dense crowds."""
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.equalizeHist(l)  # improve contrast
    lab = cv2.merge((l, a, b))
    preprocessed = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return cv2.cvtColor(preprocessed, cv2.COLOR_BGR2RGB)

def find_faces_multi_scale(image_path, scales=[0.5, 1.0, 1.5], detection_model='cnn'):
    """Detect faces at multiple scales."""
    orig_image = cv2.imread(image_path)
    if orig_image is None:
        return [], [], orig_image
    
    all_locations, all_encodings = [], []
    preprocessed_orig = preprocess_image(orig_image)
    
    for scale in scales:
        new_width = int(orig_image.shape[1] * scale)
        new_height = int(orig_image.shape[0] * scale)
        resized = cv2.resize(preprocessed_orig, (new_width, new_height))
        
        locations = face_recognition.face_locations(resized, model=detection_model)
        scaled_locations = []
        for top, right, bottom, left in locations:
            scaled_locations.append((
                int(top / scale), int(right / scale),
                int(bottom / scale), int(left / scale)
            ))
        encodings = face_recognition.face_encodings(orig_image, scaled_locations)
        all_locations.extend(scaled_locations)
        all_encodings.extend(encodings)
    
    return all_locations, all_encodings, orig_image

def non_max_suppression(boxes, overlapThresh=0.3):
    """Remove overlapping boxes using NMS."""
    if len(boxes) == 0:
        return []
    boxes = np.array(boxes)
    pick = []
    x1 = boxes[:,1]
    y1 = boxes[:,0]
    x2 = boxes[:,3]
    y2 = boxes[:,2]
    area = (x2 - x1 + 1) * (y2 - y1 +1)
    idxs = np.argsort(y2)
    
    while len(idxs) > 0:
        last = idxs[-1]
        pick.append(last)
        idxs = idxs[:-1]
        if len(idxs) == 0:
            break
        xx1 = np.maximum(x1[last], x1[idxs])
        yy1 = np.maximum(y1[last], y1[idxs])
        xx2 = np.minimum(x2[last], x2[idxs])
        yy2 = np.minimum(y2[last], y2[idxs])
        w = np.maximum(0, xx2 - xx1 +1)
        h = np.maximum(0, yy2 - yy1 +1)
        overlap = (w * h) / area[idxs]
        idxs = idxs[overlap <= overlapThresh]
    return boxes[pick].astype(int)

# --- Main Script ---

if __name__ == "__main__":
  freeze_support() 
  targets_folder = "datasets/targets"
  crowd_folder = "datasets/crowd_images"
  output_folder = "datasets/results"
  os.makedirs(output_folder, exist_ok=True)

  # 1. Encode target images in parallel
  known_encodings, known_names = [], []
  target_paths = [os.path.join(targets_folder, f) for f in os.listdir(targets_folder)
                  if f.lower().endswith((".jpg",".jpeg",".png",".webp"))]

  print(f"✅ Encoding {len(target_paths)} target images...")
  with ProcessPoolExecutor() as executor:
    results = executor.map(process_single_target_image, target_paths)

  for encoding, name in results:
    if encoding is not None:
      known_encodings.append(encoding)
      known_names.append(name)

  print(f"✅ Loaded {len(known_encodings)} target faces.")

  # 2. Process each crowd image
  for crowd_file in os.listdir(crowd_folder):
      if not crowd_file.lower().endswith((".jpg",".jpeg",".png",".webp")):
          continue

      print(f"\n🔍 Processing {crowd_file}...")
      start_time = time.time()
      crowd_path = os.path.join(crowd_folder, crowd_file)

      face_locations, face_encodings, crowd_image = find_faces_multi_scale(
          crowd_path, scales=[0.5, 1.0, 1.5], detection_model='cnn'
      )

      # Remove overlapping detections
      if len(face_locations) > 0:
          boxes = non_max_suppression(face_locations)
          # Filter encodings to match non-maximum boxes
          filtered_encodings = []
          filtered_locations = []
          for box in boxes:
              for loc, enc in zip(face_locations, face_encodings):
                  if np.all(loc == box):
                      filtered_locations.append(loc)
                      filtered_encodings.append(enc)
                      break
          face_locations = filtered_locations
          face_encodings = filtered_encodings

      recognized_targets = set()
      for i, (loc, enc) in enumerate(zip(face_locations, face_encodings)):
          top, right, bottom, left = loc
          if len(known_encodings) == 0:
              continue
          distances = face_recognition.face_distance(known_encodings, enc)
          best_idx = np.argmin(distances)
          name = "Unknown"
          confidence = 1 - distances[best_idx]
          if distances[best_idx] <= 0.5:
              name = known_names[best_idx]
              recognized_targets.add(name)
              # Crop and save
              face_crop = crowd_image[top:bottom, left:right]
              save_dir = os.path.join(output_folder, name)
              os.makedirs(save_dir, exist_ok=True)
              face_filename = f"{os.path.splitext(crowd_file)[0]}_{i}_{name}_{confidence:.2f}.jpg"
              cv2.imwrite(os.path.join(save_dir, face_filename), face_crop)

          # Annotate
          color = (0,255,0) if name != "Unknown" else (0,0,255)
          cv2.rectangle(crowd_image, (left, top), (right, bottom), color, 2)
          label = f"{name} ({confidence:.2f})"
          cv2.putText(crowd_image, label, (left, top-10),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

      annotated_path = os.path.join(output_folder, f"annotated_{crowd_file}")
      cv2.imwrite(annotated_path, crowd_image)
      elapsed = time.time() - start_time
      print(f"Found {len(face_locations)} faces. Recognized: {len(recognized_targets)}. Time: {elapsed:.2f}s")

  print("\n🎉 Processing complete.")