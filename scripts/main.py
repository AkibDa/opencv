from object_detect import FaceRecognitionSystem

def main():
  system = FaceRecognitionSystem(dataset_dir="dataset", save_dir="faces")
  system.recognise_faces(videoPath=0)  # 0 = webcam

if __name__ == "__main__":
  main()
