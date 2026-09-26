# Real-Time Face Detection and Recognition System

## 1. Project Overview

A robust, local Python AI/ML application that captures real-time webcam video streams to instantly detect and recognize faces. Built with modularity and stability in mind, it operates independently of cloud services, databases, or web backends.

## 2. Objective

This project demonstrates a streamlined computer vision pipeline—progressing from capturing an image, extracting facial feature vectors, comparing vectors, and projecting labeled overlays onto a live feed.

## 3. Technologies Used

- **Python**: Core programming language
- **OpenCV (`cv2`)**: Real-time video processing, image rendering, and hardware webcam interface
- **NumPy**: Matrix and numerical calculations
- **face-recognition**: High-level facial feature extraction (built on top of `dlib`)
- **dlib**: Deep learning models for landmark estimation and encoding

## 4. Features

- **Real-Time Detection**: Processes frames instantly using bounding box calculations.
- **Single-Reference Recognition**: Encodes a single reference photo (`images/bunny.jpeg`) and intelligently identifies if a webcam face is a match or "Unknown".
- **Multi-Face Tracking**: Simultaneously tracks and identifies an arbitrary number of faces within the frame independently.
- **Euclidean Distance Display**: Renders raw face distance calculations out to 2 decimal places in real-time.
- **Graceful Error Handling**: Bullet-proof environment and runtime guards that prevent tracebacks during frame drops, missing references, or hardware disconnects.
- **Dynamic Configuration**: Easy top-level variable configuration (tolerance, reference name, paths).

## 5. Project Structure

```text
Face Detection/
│
├── main.py                # Core orchestrator & configuration hub
├── requirements.txt       # Project dependencies
├── README.md              # Project documentation
├── pkg_resources.py       # Legacy polyfill ensuring environment stability
│
├── images/                
│   └── bunny.jpeg         # The reference identity image
│
└── src/                   
    ├── __init__.py        
    ├── face_encoder.py    # Reference image validation & encoding engine
    └── face_detector.py   # Webcam frame processing & rendering engine
```

## 6. Installation

1. Ensure you have Python installed.
2. Install the exact required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   *(Note: The `face-recognition` library relies heavily on C++ bindings via `dlib`. You may need CMake and C++ build tools installed on your operating system.)*

## 7. Running the Project

Execute the main application via terminal:

```bash
python main.py
```

## 8. Reference Image Configuration

Place the face you want to recognize at:
`images/bunny.jpeg`

*Important Constraint*: The image **must** contain exactly one clear, identifiable face. The system intentionally aborts if multiple faces or zero faces are detected in the reference photo to ensure high fidelity matching.

## 9. Recognition Tolerance Explanation

Facial recognition utilizes a Euclidean distance calculation. 
- A **lower** distance implies higher similarity.
- A **higher** distance implies lower similarity.

By default, the `FACE_TOLERANCE` is set to `0.6` in `main.py`. If the calculated distance between the reference face and the webcam face is <= 0.6, it results in a match. You can edit `main.py` to tighten or loosen this threshold.

## 10. Controls

- **Q**: Press to instantly release the webcam and safely quit the application.

## 11. Error Handling & Stability

The system differentiates between critical startup errors and localized loop errors:

- **Startup Errors**: Missing modules, missing reference images, unavailable webcam hardware—aborts cleanly with actionable CLI feedback.
- **Runtime Errors**: Glitches in an individual video frame are caught, discarded, and the application seamlessly grabs the next frame without dropping the connection.

## 12. Limitations

- Single reference-identity matching only.
- Local webcam execution only; no distributed or networked processing.
- Face Detection != Face Recognition. While detection finds the spatial coordinates of a face, recognition runs complex mathematical feature extraction. Recognition is computationally heavy.

## 13. Testing

Local hardware tests (e.g., verifying the bounding boxes are drawn on your specific webcam) **must be performed manually** by running the script.

## 14. Phase Roadmap

- **Phase 01**: Environment Scaffolding & Verification
- **Phase 02**: Face Detection & Box Rendering
- **Phase 03**: Feature Encoding & Identity Recognition
- **Phase 04**: Code Finalization, Configuration, & Performance Polish *(Current)*

## 15. Future Improvements

- Multi-identity database scaling (SQLite/Pickle).
- Threaded processing to detach GUI frame rate from model inference time.
- Implementation of a web dashboard for remote viewing.
