import sys
import os

# ==========================================
# PROJECT CONFIGURATION
# ==========================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
REFERENCE_IMAGE = os.path.join(PROJECT_ROOT, "images", "bunny.jpeg")
KNOWN_PERSON_NAME = "Bunny"
FACE_TOLERANCE = 0.6
CAMERA_INDEX = 0

def check_environment():
    """
    Verifies that the essential Python libraries are available before proceeding.
    Provides graceful degradation on missing imports.
    """
    all_passed = True
    print("Environment:")
    print("Python: OK")

    try:
        import cv2
        print("OpenCV: OK")
    except ImportError:
        print("OpenCV: FAILED")
        all_passed = False

    try:
        import numpy as np
        print("NumPy: OK")
    except ImportError:
        print("NumPy: FAILED")
        all_passed = False

    try:
        import face_recognition
        print("face-recognition: OK")
    except ImportError:
        print("face-recognition: FAILED (Module not found)")
        all_passed = False
    except Exception as e:
        print(f"face-recognition: FAILED ({e})")
        all_passed = False
    except SystemExit:
        print("face-recognition: FAILED (SystemExit, likely missing models)")
        all_passed = False
        
    if not all_passed:
        print("\nEnvironment verification FAILED. Please resolve errors.")
        sys.exit(1)
        
    print()

def main():
    """
    Main application orchestration.
    """
    print("========================================")
    print("REAL-TIME FACE RECOGNITION SYSTEM")
    print("========================================\n")
    
    # 1. Verify Environment
    check_environment()
    
    # Lazy-load hefty modules after environment check succeeds
    import cv2
    from src.face_encoder import generate_reference_encoding
    from src.face_detector import detect_faces, recognize_faces, draw_bounding_boxes
    
    # 2. Process Reference Image (Done ONCE for optimal performance)
    try:
        reference_encoding = generate_reference_encoding(REFERENCE_IMAGE)
        print("Reference image: OK")
        print("Reference face: OK")
        print("Reference encoding: OK\n")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(f"\nPlease ensure '{REFERENCE_IMAGE}' exists.")
        sys.exit(1)
    except ValueError as e:
        print(f"Error processing reference image: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error with reference image: {e}")
        sys.exit(1)
        
    # 3. Initialize Webcam
    video_capture = cv2.VideoCapture(CAMERA_INDEX)
    
    if not video_capture.isOpened():
        print(f"Webcam: FAILED (Cannot open camera index {CAMERA_INDEX})")
        sys.exit(1)
        
    print("Webcam: OK\n")
    print("Recognition started.")
    print("Controls:\nQ = Quit\n")
    
    # 4. Real-time Processing Loop
    try:
        while True:
            # Capture real-time frame
            ret, frame = video_capture.read()
            
            if not ret or frame is None:
                print("Error: Could not read frame from webcam.")
                break
                
            try:
                # Detect face locations
                face_locations = detect_faces(frame)
                
                # Recognize identities and compute distance
                recognized_data = recognize_faces(
                    frame, 
                    face_locations, 
                    reference_encoding, 
                    known_name=KNOWN_PERSON_NAME, 
                    tolerance=FACE_TOLERANCE
                )
            except Exception as e:
                # Catch localized frame errors without crashing the entire stream
                print(f"Error during face processing loop: {e}")
                continue 
                
            # Draw bounding boxes and labels
            draw_bounding_boxes(frame, recognized_data)
            
            # Display the resulting frame
            cv2.imshow("Real-Time Face Recognition", frame)
            
            # Key polling: Q to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception as e:
        print(f"\nUnexpected runtime error during webcam loop: {e}")
    finally:
        # 5. Clean Shutdown Resource Management
        video_capture.release()
        cv2.destroyAllWindows()
        print("Webcam resources released.")

if __name__ == "__main__":
    main()
