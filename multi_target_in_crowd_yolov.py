import cv2
import numpy as np
import os
import time
from ultralytics import YOLO
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
import torch

class YOLOv8FaceRecognizer:
    def __init__(self, detection_threshold=0.3, recognition_threshold=0.5):
        """
        Initialize YOLOv8 face recognizer
        
        Args:
            detection_threshold: Confidence threshold for face detection (0-1)
            recognition_threshold: Similarity threshold for face recognition (0-1)
        """
        self.detection_threshold = detection_threshold
        self.recognition_threshold = recognition_threshold
        
        print("🚀 Loading YOLOv8 Face Detection model...")
        # Load YOLOv8 face detection model (specialized for faces)
        self.face_detector = YOLO('yolov8n-face.pt')  # You can use 'yolov8m-face.pt' for better accuracy
        
        print("🚀 Loading InsightFace Recognition model...")
        # Load InsightFace for recognition (state-of-the-art)
        self.face_recognizer = FaceAnalysis(
            name='buffalo_l',  # Best available model
            providers=['CUDAExecutionProvider' if torch.cuda.is_available() else 'CPUExecutionProvider']
        )
        self.face_recognizer.prepare(ctx_id=0, det_size=(640, 640))
        
        print("✅ Models loaded successfully!")
        print(f"💻 Using device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
    
    def load_target_faces(self, targets_folder):
        """
        Load and encode target faces from folder
        
        Returns:
            dict: {name: embedding} for all target faces
        """
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
                # Read image
                img = cv2.imread(path)
                if img is None:
                    print(f"⚠️  Could not read image: {filename}")
                    continue
                
                # Get face embedding using InsightFace
                faces = self.face_recognizer.get(img)
                if not faces:
                    print(f"⚠️  No face found in: {filename}")
                    continue
                
                # Use the first face found
                name = os.path.splitext(filename)[0]
                target_embeddings[name] = faces[0].embedding
                print(f"✅ Encoded: {name}")
                
            except Exception as e:
                print(f"❌ Error processing {filename}: {str(e)}")
        
        return target_embeddings
    
    def detect_faces_yolo(self, image):
        """
        Detect faces using YOLOv8 (fast and accurate for crowds)
        
        Returns:
            list: [(x1, y1, x2, y2, confidence), ...]
        """
        # Run YOLOv8 inference
        results = self.face_detector(image, conf=self.detection_threshold, verbose=False)
        
        faces = []
        for result in results:
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()  # Bounding boxes
                confidences = result.boxes.conf.cpu().numpy()  # Confidence scores
                
                for box, conf in zip(boxes, confidences):
                    x1, y1, x2, y2 = map(int, box)
                    faces.append((x1, y1, x2, y2, conf))
        
        return faces
    
    def recognize_faces(self, image, target_embeddings):
        """
        Recognize faces in image against target embeddings
        """
        # Detect faces with YOLOv8 (fast)
        detected_faces = self.detect_faces_yolo(image)
        
        recognized_results = []
        
        for (x1, y1, x2, y2, conf) in detected_faces:
            # Extract face region
            face_region = image[y1:y2, x1:x2]
            
            if face_region.size == 0:
                continue
            
            # Get embedding for this face
            try:
                # InsightFace expects RGB
                face_rgb = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)
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
                print(f"❌ Error processing face: {str(e)}")
                recognized_results.append(((x1, y1, x2, y2), "Error", 0.0))
        
        return recognized_results
    
    def process_crowd_image(self, crowd_path, target_embeddings, output_folder):
        """
        Process a single crowd image and save results
        """
        print(f"\n🔍 Processing: {os.path.basename(crowd_path)}")
        start_time = time.time()
        
        # Read image
        image = cv2.imread(crowd_path)
        if image is None:
            print(f"❌ Could not read image: {crowd_path}")
            return set()
        
        # Recognize faces
        results = self.recognize_faces(image, target_embeddings)
        
        recognized_names = set()
        
        # Draw results and save cropped faces
        for (x1, y1, x2, y2), name, score in results:
            # Set color and label
            if name != "Unknown" and score > 0:
                color = (0, 255, 0)  # Green for known
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
                color = (0, 0, 255)  # Red for unknown
                label = "Unknown"
            
            # Draw bounding box and label
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
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
    OUTPUT_FOLDER = "datasets/results_yolo"
    
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    # Initialize recognizer
    recognizer = YOLOv8FaceRecognizer(
        detection_threshold=0.3,    # Lower = more faces detected (but more false positives)
        recognition_threshold=0.5   # Higher = more strict matching
    )
    
    # Load target faces
    target_embeddings = recognizer.load_target_faces(TARGETS_FOLDER)
    
    if not target_embeddings:
        print("❌ No target faces loaded. Please add images to the targets folder.")
        return
    
    print(f"\n🎯 Loaded {len(target_embeddings)} target faces for recognition")
    
    # Process all crowd images
    crowd_files = [f for f in os.listdir(CROWD_FOLDER) 
                  if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))]
    
    if not crowd_files:
        print("❌ No crowd images found.")
        return
    
    total_start = time.time()
    
    for crowd_file in crowd_files:
        crowd_path = os.path.join(CROWD_FOLDER, crowd_file)
        recognizer.process_crowd_image(crowd_path, target_embeddings, OUTPUT_FOLDER)
    
    total_time = time.time() - total_start
    print(f"\n🎉 All images processed in {total_time:.2f} seconds!")
    print(f"📁 Results saved in: {OUTPUT_FOLDER}")

if __name__ == "__main__":
    main()