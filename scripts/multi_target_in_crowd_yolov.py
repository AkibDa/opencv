import cv2
import numpy as np
import os
import time
from ultralytics import YOLO
from insightface.app import FaceAnalysis
import torch

class MacFaceRecognizer:
    def __init__(self, detection_threshold=0.4, recognition_threshold=0.5):
        """
        Initialize face recognizer optimized for macOS CPU
        """
        self.detection_threshold = detection_threshold
        self.recognition_threshold = recognition_threshold
        
        print("🍎 Initializing macOS-optimized face recognition...")
        
        # Check if we can use Metal Performance Shaders (Apple Silicon)
        self.device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        print(f"💻 Using device: {self.device.upper()}")
        
        print("🚀 Loading YOLOv8 Face Detection model...")
        # Use smaller model for CPU
        self.face_detector = YOLO('yolov8n-face.pt')
        
        print("🚀 Loading InsightFace Recognition model...")
        # Configure for CPU optimization
        self.face_recognizer = FaceAnalysis(
            name='buffalo_l',
            providers=['CPUExecutionProvider']  # Force CPU only
        )
        self.face_recognizer.prepare(ctx_id=-1, det_size=(320, 320))  # Smaller size for CPU
        
        print("✅ Models loaded successfully!")
    
    def load_target_faces(self, targets_folder):
        """Load and encode target faces from folder"""
        target_embeddings = {}
        
        if not os.path.exists(targets_folder):
            print(f"⚠️  Targets folder '{targets_folder}' not found. Creating empty database.")
            return target_embeddings
        
        target_files = [f for f in os.listdir(targets_folder) 
                       if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))]
        
        if not target_files:
            print("⚠️  No target images found in folder.")
            return target_embeddings
        
        print(f"📸 Encoding {len(target_files)} target faces...")
        
        for filename in target_files:
            path = os.path.join(targets_folder, filename)
            try:
                img = cv2.imread(path)
                if img is None:
                    print(f"⚠️  Could not read image: {filename}")
                    continue
                
                # Resize for faster processing on CPU
                img_small = cv2.resize(img, (320, 320))
                faces = self.face_recognizer.get(img_small)
                
                if not faces:
                    print(f"⚠️  No face found in: {filename}")
                    continue
                
                name = os.path.splitext(filename)[0]
                target_embeddings[name] = faces[0].embedding
                print(f"✅ Encoded: {name}")
                
            except Exception as e:
                print(f"❌ Error processing {filename}: {str(e)}")
        
        return target_embeddings
    
    def detect_faces_yolo(self, image):
        """Detect faces using YOLOv8 with CPU optimization"""
        # Downscale image for faster processing on CPU
        height, width = image.shape[:2]
        scale_factor = 0.6  # Reduce size for CPU
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        small_image = cv2.resize(image, (new_width, new_height))
        
        results = self.face_detector(small_image, conf=self.detection_threshold, verbose=False)
        
        faces = []
        for result in results:
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                
                for box, conf in zip(boxes, confidences):
                    # Scale coordinates back to original size
                    x1, y1, x2, y2 = map(int, box / scale_factor)
                    faces.append((x1, y1, x2, y2, conf))
        
        return faces
    
    def recognize_faces(self, image, target_embeddings):
        """Recognize faces in image against target embeddings"""
        detected_faces = self.detect_faces_yolo(image)
        recognized_results = []
        
        for (x1, y1, x2, y2, conf) in detected_faces:
            # Extract face region with padding
            padding = 10
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(image.shape[1], x2 + padding)
            y2 = min(image.shape[0], y2 + padding)
            
            face_region = image[y1:y2, x1:x2]
            
            if face_region.size == 0:
                continue
            
            try:
                # Resize face for consistent processing
                face_resized = cv2.resize(face_region, (112, 112))
                face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
                
                current_face = self.face_recognizer.get(face_rgb)
                
                if not current_face:
                    recognized_results.append(((x1, y1, x2, y2), "Unknown", 0.0))
                    continue
                
                current_embedding = current_face[0].embedding
                
                # Find best match
                best_match = "Unknown"
                best_score = 0.0
                
                for name, target_embedding in target_embeddings.items():
                    similarity = np.dot(current_embedding, target_embedding)
                    if similarity > best_score and similarity > self.recognition_threshold:
                        best_score = similarity
                        best_match = name
                
                recognized_results.append(((x1, y1, x2, y2), best_match, best_score))
                
            except Exception as e:
                recognized_results.append(((x1, y1, x2, y2), "Error", 0.0))
        
        return recognized_results
    
    def process_crowd_image(self, crowd_path, target_embeddings, output_folder):
        """Process a single crowd image with macOS optimizations"""
        print(f"\n🔍 Processing: {os.path.basename(crowd_path)}")
        start_time = time.time()
        
        image = cv2.imread(crowd_path)
        if image is None:
            print(f"❌ Could not read image: {crowd_path}")
            return set()
        
        # Downscale large images for CPU processing
        height, width = image.shape[:2]
        max_dimension = 1200  # Limit size for CPU
        if max(height, width) > max_dimension:
            scale = max_dimension / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = cv2.resize(image, (new_width, new_height))
        
        results = self.recognize_faces(image, target_embeddings)
        recognized_names = set()
        
        # Draw results
        for (x1, y1, x2, y2), name, score in results:
            if name != "Unknown" and score > 0:
                color = (0, 255, 0)
                label = f"{name} ({score:.2f})"
                recognized_names.add(name)
                
                # Save cropped face
                face_crop = image[y1:y2, x1:x2]
                if face_crop.size > 0:
                    save_dir = os.path.join(output_folder, name)
                    os.makedirs(save_dir, exist_ok=True)
                    timestamp = int(time.time() * 1000)
                    save_path = os.path.join(save_dir, f"{name}_{timestamp}.jpg")
                    cv2.imwrite(save_path, face_crop)
            else:
                color = (0, 0, 255)
                label = "Unknown"
            
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Save annotated image
        annotated_path = os.path.join(output_folder, f"annotated_{os.path.basename(crowd_path)}")
        cv2.imwrite(annotated_path, image)
        
        elapsed = time.time() - start_time
        print(f"✅ Found {len(results)} faces, recognized {len(recognized_names)} targets")
        print(f"⏱ Time: {elapsed:.2f} seconds")
        
        return recognized_names

def main():
    # Configuration
    TARGETS_FOLDER = "datasets/targets"
    CROWD_FOLDER = "datasets/crowd_images"
    OUTPUT_FOLDER = "datasets/results_mac"
    
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    # Initialize recognizer with macOS-optimized settings
    recognizer = MacFaceRecognizer(
        detection_threshold=0.4,    # Slightly higher to reduce false positives
        recognition_threshold=0.55   # Slightly higher for better accuracy
    )
    
    # Load target faces
    target_embeddings = recognizer.load_target_faces(TARGETS_FOLDER)
    
    if not target_embeddings:
        print("❌ No target faces loaded. Please add images to the targets folder.")
        print("💡 You can use your extraction script to get faces from crowd images first.")
        return
    
    print(f"\n🎯 Loaded {len(target_embeddings)} target faces for recognition")
    
    # Process crowd images
    crowd_files = [f for f in os.listdir(CROWD_FOLDER) 
                  if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))]
    
    if not crowd_files:
        print("❌ No crowd images found.")
        return
    
    total_start = time.time()
    
    for i, crowd_file in enumerate(crowd_files):
        crowd_path = os.path.join(CROWD_FOLDER, crowd_file)
        print(f"\n📊 Processing image {i+1}/{len(crowd_files)}")
        recognizer.process_crowd_image(crowd_path, target_embeddings, OUTPUT_FOLDER)
    
    total_time = time.time() - total_start
    print(f"\n🎉 All {len(crowd_files)} images processed in {total_time:.2f} seconds!")
    print(f"📁 Results saved in: {OUTPUT_FOLDER}")

if __name__ == "__main__":
    main()