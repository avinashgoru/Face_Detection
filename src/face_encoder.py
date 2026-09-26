import os
import face_recognition

def generate_reference_encoding(image_path):
    """
    Loads an image, detects faces, and generates a face encoding.
    Ensures exactly one face is present in the reference image.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Reference image not found at: {image_path}")
        
    try:
        # Load the image file
        image = face_recognition.load_image_file(image_path)
    except Exception as e:
        raise ValueError(f"Failed to load image file {image_path}: {e}")

    # Detect face locations
    face_locations = face_recognition.face_locations(image)
    
    if not face_locations:
        raise ValueError(f"No faces detected in the reference image: {image_path}")
        
    if len(face_locations) > 1:
        raise ValueError(f"Multiple faces ({len(face_locations)}) detected in {image_path}. Please use an image with exactly one identifiable face.")
        
    # Generate face encoding
    try:
        face_encodings = face_recognition.face_encodings(image, face_locations)
        if not face_encodings:
            raise ValueError("Failed to generate face encoding.")
        return face_encodings[0]
    except Exception as e:
        raise RuntimeError(f"Error generating face encoding: {e}")
