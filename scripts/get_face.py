import face_recognition
import cv2
import os
import numpy as np

def extract_faces_from_crowd(crowd_image_path, output_dir, min_face_size=100, quality_threshold=0.8):
    """
    Extract high-quality faces from a crowd image to build a target database.
    
    Args:
        crowd_image_path: Path to the crowd image
        output_dir: Directory to save extracted faces
        min_face_size: Minimum face size in pixels to keep (avoids tiny faces)
        quality_threshold: Minimum confidence score to consider a face good quality
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load the image
    print("📷 Loading crowd image...")
    image = face_recognition.load_image_file(crowd_image_path)
    original_height, original_width = image.shape[:2]
    
    # Calculate scale factor to ensure decent resolution
    # We want the larger dimension to be around 2000px for good CNN detection
    max_dimension = max(original_height, original_width)
    scale_factor = min(2.0, 2000 / max_dimension)  # Don't scale up more than 2x
    
    print(f"🔄 Scaling image by factor {scale_factor:.2f}...")
    new_width = int(original_width * scale_factor)
    new_height = int(original_height * scale_factor)
    image_scaled = cv2.resize(image, (new_width, new_height))
    
    # Detect faces using CNN (most accurate)
    print("🔍 Detecting faces with CNN model (this may take a while)...")
    face_locations = face_recognition.face_locations(image_scaled, model="cnn")
    face_landmarks_list = face_recognition.face_landmarks(image_scaled, face_locations)
    
    print(f"✅ Found {len(face_locations)} potential face(s)")
    
    # Convert to BGR for OpenCV operations
    image_cv = cv2.cvtColor(image_scaled, cv2.COLOR_RGB2BGR)
    
    face_count = 0
    saved_count = 0
    
    for i, ((top, right, bottom, left), landmarks) in enumerate(zip(face_locations, face_landmarks_list)):
        # Calculate face dimensions
        face_height = bottom - top
        face_width = right - left
        
        # Skip faces that are too small
        if face_height < min_face_size or face_width < min_face_size:
            print(f"  ⏩ Skipping face {i+1}: Too small ({face_width}x{face_height})")
            continue
        
        # Extract the face region
        face_crop = image_cv[top:bottom, left:right]
        
        # Calculate quality score based on face proportions and landmarks
        quality_score = calculate_face_quality(face_crop, landmarks)
        
        if quality_score < quality_threshold:
            print(f"  ⏩ Skipping face {i+1}: Low quality (score: {quality_score:.2f})")
            continue
        
        # Enhance the face image
        enhanced_face = enhance_face_image(face_crop)
        
        # Save with quality score in filename
        face_count += 1
        filename = f"target_{face_count:03d}_q{quality_score:.2f}.jpg"
        save_path = os.path.join(output_dir, filename)
        cv2.imwrite(save_path, enhanced_face)
        saved_count += 1
        
        print(f"  💾 Saved face {i+1}: {filename} ({face_width}x{face_height}, quality: {quality_score:.2f})")
    
    print(f"\n🎉 Extraction complete! Saved {saved_count} high-quality faces out of {len(face_locations)} detected.")
    return saved_count

def calculate_face_quality(face_image, landmarks):
    """
    Calculate a quality score for the face based on various factors.
    Returns a score between 0 and 1.
    """
    height, width = face_image.shape[:2]
    
    # 1. Aspect ratio score (ideal is around 0.75-0.85)
    aspect_ratio = height / width
    aspect_score = 1.0 - min(abs(aspect_ratio - 0.8) / 0.2, 1.0)
    
    # 2. Sharpness score (using Laplacian variance)
    gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    sharpness_score = min(sharpness / 100.0, 1.0)  # Normalize
    
    # 3. Brightness score (avoid too dark or too bright)
    brightness = np.mean(gray)
    brightness_score = 1.0 - min(abs(brightness - 127) / 127, 1.0)
    
    # 4. Check if key landmarks are detected
    landmark_score = 1.0
    required_landmarks = ['left_eye', 'right_eye', 'nose_bridge']
    for landmark in required_landmarks:
        if landmark not in landmarks or len(landmarks[landmark]) < 2:
            landmark_score *= 0.7
    
    # 5. Eye alignment score (rough estimate)
    eye_alignment_score = 1.0
    if 'left_eye' in landmarks and 'right_eye' in landmarks:
        left_eye_center = np.mean(landmarks['left_eye'], axis=0)
        right_eye_center = np.mean(landmarks['right_eye'], axis=0)
        eye_distance = np.linalg.norm(left_eye_center - right_eye_center)
        # Ideal eye distance is about 1/3 of face width
        ideal_distance = width / 3
        eye_alignment_score = 1.0 - min(abs(eye_distance - ideal_distance) / ideal_distance, 1.0)
    
    # Combine scores with weights
    total_score = (aspect_score * 0.2 + 
                  sharpness_score * 0.3 + 
                  brightness_score * 0.2 + 
                  landmark_score * 0.2 +
                  eye_alignment_score * 0.1)
    
    return total_score

def enhance_face_image(face_image):
    """
    Apply basic enhancements to improve face image quality.
    """
    # Convert to LAB color space for better contrast handling
    lab = cv2.cvtColor(face_image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L channel for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    
    # Merge enhanced L channel with original A and B channels
    lab_enhanced = cv2.merge((l_enhanced, a, b))
    enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    
    # Mild sharpening
    kernel = np.array([[-1, -1, -1],
                       [-1, 9, -1],
                       [-1, -1, -1]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)
    
    return sharpened

if __name__ == "__main__":
    # Configuration
    crowd_image_path = "datasets/crowd_images/IMG_8093.jpg"  # Your crowd image
    output_dir = "datasets/detected_faces"  # Where to save extracted faces
    
    print("🚀 Starting high-quality face extraction from crowd image...")
    print("=" * 60)
    
    extracted_count = extract_faces_from_crowd(
        crowd_image_path, 
        output_dir,
        min_face_size=80,      # Minimum face size in pixels
        quality_threshold=0.6  # Minimum quality score to keep
    )
    
    print("=" * 60)
    if extracted_count > 0:
        print(f"✅ Successfully extracted {extracted_count} high-quality target faces!")
        print(f"📁 They are saved in: {output_dir}")
        print("\n💡 Next steps:")
        print("1. Review the extracted faces in the target folder")
        print("2. Rename files to match person names (e.g., 'john_doe_q0.85.jpg')")
        print("3. Remove any low-quality or incorrect detections")
        print("4. Run your face recognition script with this new target database")
    else:
        print("❌ No suitable faces were extracted. Try with a different image or adjust parameters.")