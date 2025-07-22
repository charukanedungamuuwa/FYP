import cv2
import torch
import base64
import collections
import time
from ultralytics import YOLO
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from gtts import gTTS
import os
import numpy as np
from io import BytesIO
from pydantic import BaseModel

app = FastAPI()

# CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to frontend domain if deployed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load YOLO models
try:
    object_model = YOLO("yolo_models/object_model.pt")
    touch_model = YOLO("yolo_models/touch_model.pt")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Models loaded successfully on device: {device}")
    print("Object model class names:", object_model.names)
    print("Touch model class names:", touch_model.names)
except Exception as e:
    print(f"Error loading models: {e}")
    object_model = None
    touch_model = None

# Object introductions - enhanced like in your original code
# introductions = {
#     "cube": "You are holding a  cube. It feels the same on all sides. It has six flat square faces, twelve straight edges, and eight corners called vertices. All sides are equal in size.",
#     "cuboid": "You are holding a cuboid. It feels like a box. It has six flat faces, shaped like rectangles, twelve straight edges, and eight vertices. Some sides are longer than others.",
#     "cone": "You are holding a cone. Feel the round base at the bottom and a smooth curved surface going up to a single sharp point called the apex. It has one edge and one vertex.",
#     "tetrahedrone": "You are holding a tetrahedron. It feels like a pyramid with a triangle base. It has four triangular faces, six straight edges, and four vertices.",
#     "cylinder": "You are holding a cylinder. It has a round base and a round top, connected by a smooth curved surface. You will not feel any corners or vertices.",
#     "prism": "You are holding a triangular prism. It has two triangle-shaped ends and three rectangle-shaped sides. It has nine edges and six vertices."
# }

introductions = {
    "cube": " It feels the same on all sides. It has six flat square faces, twelve straight edges, and eight corners called vertices. All sides are equal in size.",
    "cuboid": "It feels like a box. It has six flat faces, shaped like rectangles, twelve straight edges, and eight vertices. Some sides are longer than others.",
    "cone": "Feel the round base at the bottom and a smooth curved surface going up to a single sharp point called the apex. It has one edge and one vertex.",
    "tetrahedrone": "It feels like a pyramid with a triangle base. It has four triangular faces, six straight edges, and four vertices.",
    "cylinder": " It has a round base and a round top, connected by a smooth curved surface. You will not feel any corners or vertices.",
    "prism": "It has two triangle-shaped ends and three rectangle-shaped sides. It has nine edges and six vertices."
}


# Session storage for rotation-based detection
rotation_sessions = {}

# Feature detection state
is_processing_feature = False

class RotationSession(BaseModel):
    session_id: str
    label_counts: dict = {}
    frame_count: int = 0
    target_frames: int = 200
    detection_threshold: int = 101

def speak_as_base64(text: str) -> str:
    """Convert text to speech and return as base64 encoded audio"""
    try:
        tts = gTTS(text=text)
        buf = BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        audio_bytes = buf.read()
        return base64.b64encode(audio_bytes).decode("utf-8")
    except Exception as e:
        print(f"TTS Error: {e}")
        return ""

@app.get("/")
async def root():
    return {"message": "3D Object Teaching API is running!"}

# @app.get("/initial-instruction")
# async def get_initial_instruction():
#     """Get initial instruction audio for when app loads"""
#     audio = speak_as_base64("Press F to start object detection")
#     return JSONResponse({
#         "message": "Press F to start object detection",
#         "audio": audio
#     })

@app.post("/start-rotation-detection/")
async def start_rotation_detection(session_data: dict):
    """Initialize rotation-based object detection session"""
    session_id = session_data.get("session_id", str(int(time.time())))
    rotation_sessions[session_id] = {
        "label_counts": collections.Counter(),
        "frame_count": 0,
        "target_frames": 200,
        "detection_threshold": 101
    }
    
    audio = speak_as_base64(" please hold and slowly rotate the object for analysis.")
    return JSONResponse({
        "session_id": session_id,
        "message": "Rotation detection started",
        "audio": audio
    })

@app.post("/detect-object-rotation/")
async def detect_object_rotation(file: UploadFile = File(...)):
    """Process single frame during rotation detection"""
    if not object_model:
        raise HTTPException(status_code=500, detail="Object model not loaded")
    
    # Get session_id from form data or headers
    session_id = "default"  # You can modify this to get from request
    
    if session_id not in rotation_sessions:
        rotation_sessions[session_id] = {
            "label_counts": collections.Counter(),
            "frame_count": 0,
            "target_frames": 200,
            "detection_threshold": 101
        }
    
    session = rotation_sessions[session_id]
    
    try:
        contents = await file.read()
        img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
        
        results = object_model.predict(img, imgsz=640, conf=0.5, device=device)[0]
        
        detected_label = None
        bounding_box = None
        
        if results.boxes:
            box = results.boxes[0]
            cls_id = int(box.cls[0])
            label_raw = object_model.names[cls_id]
            label = label_raw.strip().lower().replace(" ", "_")
            session["label_counts"][label] += 1
            detected_label = label
            
            # Get bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            bounding_box = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
        
        session["frame_count"] += 1
        
        # Check if we have enough frames for confident detection
        if session["frame_count"] >= session["target_frames"]:
            if session["label_counts"]:
                most_common_label, count = session["label_counts"].most_common(1)[0]
                if count >= session["detection_threshold"]:
                    intro = introductions.get(most_common_label, f"This is a {most_common_label.replace('_', ' ')}.")
                    full_message = f"You are holding a {most_common_label.replace('_', ' ')}. {intro}"
                    audio_b64 = speak_as_base64(full_message)
                    
                    # Clean up session
                    del rotation_sessions[session_id]
                    
                    return JSONResponse({
                        "object": most_common_label,
                        "description": intro,
                        "full_message": full_message,
                        "confidence": count,
                        "total_votes": session["frame_count"],
                        "detection_complete": True,
                        "audio": audio_b64,
                        "next_step": "Press T to start feature detection"
                    })
                else:
                    # Not confident enough
                    del rotation_sessions[session_id]
                    audio_b64 = speak_as_base64("Detection not confident. Please rotate the object again.")
                    return JSONResponse({
                        "detection_complete": False,
                        "error": "Detection not confident enough",
                        "audio": audio_b64
                    })
            else:
                del rotation_sessions[session_id]
                audio_b64 = speak_as_base64("No object detected. Please try again.")
                return JSONResponse({
                    "detection_complete": False,
                    "error": "No object detected",
                    "audio": audio_b64
                })
        
        # Still collecting frames
        return JSONResponse({
            "detection_complete": False,
            "frame_count": session["frame_count"],
            "target_frames": session["target_frames"],
            "current_detection": detected_label,
            "bounding_box": bounding_box,
            "progress": session["frame_count"] / session["target_frames"]
        })
        
    except Exception as e:
        print(f"Detection error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/detect-object/")
async def detect_object(file: UploadFile = File(...)):
    """Simple object detection (original functionality)"""
    if not object_model:
        raise HTTPException(status_code=500, detail="Object model not loaded")
    
    try:
        contents = await file.read()
        img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
        
        label_counts = collections.Counter()
        for _ in range(40):  # Simulate multiple frames
            results = object_model.predict(img, imgsz=640, conf=0.5, device=device)[0]
            if results.boxes:
                cls_id = int(results.boxes[0].cls[0])
                label = object_model.names[cls_id].strip().lower().replace(" ", "_")
                label_counts[label] += 1
        
        if label_counts:
            detected_label, count = label_counts.most_common(1)[0]
            intro = introductions.get(detected_label, f"This is a {detected_label}.")
            full_message = f"You are holding a {detected_label.replace('_', ' ')}. {intro}"
            audio_b64 = speak_as_base64(full_message)
            return JSONResponse({
                "object": detected_label,
                "description": intro,
                "full_message": full_message,
                "audio": audio_b64
            })
        else:
            return JSONResponse(status_code=400, content={"error": "No object confidently detected"})
    except Exception as e:
        print(f"Detection error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/detect-feature/")
async def detect_feature(file: UploadFile = File(...)):
    """Detect touched features on the object"""
    global is_processing_feature
    
    if not touch_model:
        raise HTTPException(status_code=500, detail="Touch model not loaded")
    
    try:
        # If currently processing a feature, return no detection
        if is_processing_feature:
            return JSONResponse({"feature": None, "is_processing": True})
            
        contents = await file.read()
        img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
        
        results = touch_model.predict(img, imgsz=640, conf=0.3, device=device)[0]
        if results.boxes:
            cls_id = int(results.boxes[0].cls[0])
            label = touch_model.names[cls_id].strip().lower().replace("_", " ")
            
            # Get bounding box for visualization
            box = results.boxes[0]
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            
            return JSONResponse({
                "feature": label,
                "bounding_box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                "confidence": float(box.conf[0]),
                "is_processing": False
            })
        return JSONResponse({"feature": None, "is_processing": False})
    except Exception as e:
        print(f"Feature detection error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/start-feature-announcement/")
async def start_feature_announcement():
    """Start the feature announcement process"""
    global is_processing_feature
    is_processing_feature = True
    return JSONResponse({"status": "started"})

@app.post("/end-feature-announcement/")
async def end_feature_announcement():
    """End the feature announcement process"""
    global is_processing_feature
    is_processing_feature = False
    return JSONResponse({"status": "ended"})

@app.post("/speak-feature/")
async def speak_feature(data: dict):
    """Generate audio for feature detection announcements"""
    feature = data.get("feature", "")
    is_next_instruction = data.get("is_next_instruction", False)
    
    try:
        if is_next_instruction:
            text = "Please move to the next feature"
        else:
            text = f"You touched a {feature}"
        
        audio_b64 = speak_as_base64(text)
        return JSONResponse({
            "audio": audio_b64,
            "text": text
        })
    except Exception as e:
        print(f"Feature speech error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/models/status")
async def get_models_status():
    """Check if models are loaded properly"""
    return JSONResponse({
        "object_model_loaded": object_model is not None,
        "touch_model_loaded": touch_model is not None,
        "device": device if 'device' in globals() else "unknown",
        "object_classes": object_model.names if object_model else {},
        "touch_classes": touch_model.names if touch_model else {}
    })

@app.post("/speak/")
async def speak_text(data: dict):
    """Convert any text to speech"""
    text = data.get("text", "")
    if not text:
        return JSONResponse(status_code=400, content={"error": "No text provided"})
    
    audio_b64 = speak_as_base64(text)
    return JSONResponse({"audio": audio_b64})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


