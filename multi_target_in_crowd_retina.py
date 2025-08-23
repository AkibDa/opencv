import cv2
import os
import numpy as np
from retinaface import RetinaFace

def extract_faces_retina(crowd_image_path, output_dir, min_face_size=50):
    """
    Extract faces from a crowd image using RetinaFace.
    
    Args:
        crowd_image_path: Path to crowd image
        output_dir: Folder to save extracted faces
        min_face_size: Minimum width/height in pixels
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load image
    print("📷 Loading crowd image...")
    image = cv2.imread(crowd_image_path)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    print("🔍 Detecting faces with RetinaFace...")
    faces = RetinaFace.detect_faces(crowd_image_path)
    
    if not faces:
        print("❌ No faces detected!")
        return 0
    
    saved_count = 0
    for i, (face_id, face_data) in enumerate(faces.items(), start=1):
        x1, y1, x2, y2 = face_data['facial_area']
        face_width = x2 - x1
        face_height = y2 - y1
        
        if face_width < min_face_size or face_height < min_face_size:
            print(f"  ⏩ Skipping face {i}: Too small ({face_width}x{face_height})")
            continue
        
        face_crop = image[y1:y2, x1:x2]
        # Optional: enhance contrast/sharpness
        enhanced_face = enhance_face_image(face_crop)
        
        filename = f"face_{saved_count+1:03d}.jpg"
        cv2.imwrite(os.path.join(output_dir, filename), enhanced_face)
        print(f"  💾 Saved face {i}: {filename} ({face_width}x{face_height})")
        saved_count += 1
    
    print(f"\n🎉 Extraction complete! Saved {saved_count} faces.")
    return saved_count

def enhance_face_image(face_image):
    """Basic enhancement for face images."""
    lab = cv2.cvtColor(face_image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    lab_enhanced = cv2.merge((l_enhanced, a, b))
    enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    
    # Mild sharpening
    kernel = np.array([[-1, -1, -1],
                       [-1, 9, -1],
                       [-1, -1, -1]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)
    
    return sharpened

if __name__ == "__main__":
    crowd_image_path = "datasets/crowd_images/IMG_8093.jpg"
    output_dir = "datasets/targets_retina"
    
    print("🚀 Starting RetinaFace extraction...")
    print("=" * 50)
    
    count = extract_faces_retina(crowd_image_path, output_dir)
    
    print("=" * 50)
    print(f"✅ Total faces saved: {count}")
