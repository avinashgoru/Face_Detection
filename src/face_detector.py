import cv2
import face_recognition

def detect_faces(frame):
    """
    Receives an OpenCV frame (BGR), converts it to RGB correctly, 
    and returns detected face locations using face_recognition.
    """
    if frame is None:
        return []
        
    # OpenCV uses BGR by default, face_recognition requires RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detect face locations in the frame
    face_locations = face_recognition.face_locations(rgb_frame)
    
    return face_locations


def recognize_faces(frame, face_locations, known_encoding, known_name="Known", tolerance=0.6):
    """
    Generates encodings for the detected faces, compares them 
    against the known reference encoding, and calculates distance.
    Returns a list of dictionaries with identity and distance details.
    """
    if not face_locations or frame is None:
        return []
        
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    
    recognized_data = []
    
    for face_encoding, location in zip(face_encodings, face_locations):
        # Calculate face distance
        face_distances = face_recognition.face_distance([known_encoding], face_encoding)
        distance = face_distances[0]
        
        # Compare based on configured tolerance
        matches = face_recognition.compare_faces([known_encoding], face_encoding, tolerance=tolerance)
        match = matches[0]
        
        name = known_name if match else "Unknown"
        
        recognized_data.append({
            "location": location,
            "name": name,
            "distance": distance,
            "match": match
        })
        
    return recognized_data


def draw_bounding_boxes(frame, recognized_data):
    """
    Draws a highly visible rectangle around each detected face,
    along with its recognized name and calculated face distance.
    Color coding is based on whether it is a match.
    """
    for data in recognized_data:
        top, right, bottom, left = data["location"]
        name = data["name"]
        distance = data["distance"]
        match = data["match"]
        
        # Determine colors (Green for known match, Red for Unknown)
        box_color = (0, 255, 0) if match else (0, 0, 255)
        
        # Draw a simple box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), box_color, 2)
        
        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 30), (right, bottom), box_color, cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.7, (255, 255, 255), 1)
        
        # Draw distance below the name box
        dist_label = f"Distance: {distance:.2f}"
        cv2.rectangle(frame, (left, bottom), (right, bottom + 25), (0, 0, 0), cv2.FILLED)
        cv2.putText(frame, dist_label, (left + 6, bottom + 18), font, 0.5, (255, 255, 255), 1)
