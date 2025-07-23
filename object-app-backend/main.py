import cv2
import torch
import base64
import collections
import time
from ultralytics import YOLO
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
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

# Object introductions in both languages
introductions = {
    "en": {
        "cube": "You are holding a cube. It feels the same on all sides. It has six flat square faces, twelve straight edges, and eight corners called vertices. All sides are equal in size.",
        "cuboid": "You are holding a cuboid. It feels like a box. It has six flat faces, shaped like rectangles, twelve straight edges, and eight vertices. Some sides are longer than others.",
        "cone": "You are holding a cone. Feel the round base at the bottom and a smooth curved surface going up to a single sharp point called the apex. It has one edge and one vertex.",
        "tetrahedrone": "You are holding a tetrahedron. It feels like a pyramid with a triangle base. It has four triangular faces, six straight edges, and four vertices.",
        "cylinder": "You are holding a cylinder. It has a round base and a round top, connected by a smooth curved surface. You will not feel any corners or vertices.",
        "prism": "You are holding a triangular prism. It has two triangle-shaped ends and three rectangle-shaped sides. It has nine edges and six vertices."
    },
    "si": {
        "cube": "ඔබ අතැතිව ඇත්තේ ඝනකයකි. එය සෑම පැත්තකින්ම එක හා සමාන ය. එයට සමචතුරස්‍රාකාර පැති හයක්, සෘජු දාර දොළහක් සහ සිරස් අට ක් ඇත. සියලුම පැති එක හා සමාන ප්‍රමාණයේ වේ.",
        "cuboid": "ඔබ අතැතිව ඇත්තේ දීර්ඝ ඝනකයකි. එය පෙට්ටියක් මෙන් දැනේ. එයට තිරස්කෝණාස්‍රාකාර පැති හයක්, සෘජු දාර දොළහක් සහ සිරස් අටක් ඇත. සමහර පැති අනෙක් ඒවාට වඩා දිග ය.",
        "cone": "ඔබ අතැතිව ඇත්තේ ශංකුවකි. පහළ වෘත්තාකාර පාදය සහ ඉහළට යන තුඩ දක්වා සිනිඳු වක්‍ර පෘෂ්ඨය හඳුනා ගන්න. එයට එක් දාරයක් සහ එක් සිරසක් ඇත.",
        "tetrahedrone": "ඔබ අතැතිව ඇත්තේ චතුස්තලයකි. එය ත්‍රිකෝණාකාර පාදයක් සහිත පිරමිඩයක් මෙන් දැනේ. එයට ත්‍රිකෝණාකාර මුහුණත් හතරක්, සෘජු දාර හයක් සහ සිරස් හතරක් ඇත.",
        "cylinder": "ඔබ අතැතිව ඇත්තේ සිලින්ඩරයකි. එයට වෘත්තාකාර පාදයක් සහ මුදුනක් ඇති අතර ඒවා සිනිඳු වක්‍ර පෘෂ්ඨයකින් සම්බන්ධ වේ. එයට කිසිදු කෙළවරක් හෝ සිරස් නොමැත.",
        "prism": "ඔබ අතැතිව ඇත්තේ ත්‍රිකෝණාකාර ප්‍රිස්මයකි. එයට ත්‍රිකෝණාකාර කෙළවර දෙකක් සහ තිරස්කෝණාස්‍රාකාර පැති තුනක් ඇත. එයට දාර නවයක් සහ සිරස් හයක් ඇත."
    }
}


# Session storage for rotation-based detection
rotation_sessions = {}

# Feature detection state
is_processing_feature = False

# Feature translations
feature_translations = {
    "en": {
        "touching_circle_face": "circle face",
        "touching_curved_edge": "curved edge",
        "touching_curved_surface": "curved surface",
        "touching_rectangular_face": "rectangular face",
        "touching_square_face": "square face",
        "touching_straight_edge": "straight edge",
        "touching_triangular_face": "triangular face",
        "touching_vertex": "vertex"
    },
    "si": {
        "touching_circle_face": "වෘත්තාකාර මුහුණත",
        "touching_curved_edge": "වක්‍ර දාරය",
        "touching_curved_surface": "වක්‍ර පෘෂ්ඨය",
        "touching_rectangular_face": "තිරස්කෝණාස්‍රාකාර මුහුණත",
        "touching_square_face": "සමචතුරස්‍රාකාර මුහුණත",
        "touching_straight_edge": "සෘජු දාරය",
        "touching_triangular_face": "ත්‍රිකෝණාකාර මුහුණත",
        "touching_vertex": "ශීර්ෂය"
    }
}

class RotationSession(BaseModel):
    session_id: str
    label_counts: dict = {}
    frame_count: int = 0
    target_frames: int = 50
    detection_threshold: int = 26

def speak_as_base64(text: str, language: str = 'en') -> str:
    """Convert text to speech and return as base64 encoded audio"""
    try:
        tts = gTTS(text=text, lang=language)
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
async def detect_object_rotation(file: UploadFile = File(...), language: str = Form("en")):
    """Process single frame during rotation detection"""
    if not object_model:
        raise HTTPException(status_code=500, detail="Object model not loaded")
    
    # Get session_id from form data or headers
    session_id = "default"
    
    if session_id not in rotation_sessions:
        rotation_sessions[session_id] = {
            "label_counts": collections.Counter(),
            "frame_count": 0,
            "target_frames": 200,
            "detection_threshold": 101,
            "language": language
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
                    # Get description in correct language
                    intro = introductions[language].get(
                        most_common_label,
                        f"This is a {most_common_label.replace('_', ' ')}." if language == "en" else f"මෙය {most_common_label.replace('_', ' ')} එකකි."
                    )
                    
                    # Generate audio in correct language
                    audio_b64 = speak_as_base64(intro, language)
                    
                    # Clean up session
                    del rotation_sessions[session_id]
                    
                    # Return response with language-specific messages
                    next_step = "Press T to start feature detection" if language == "en" else "ලක්ෂණ හඳුනා ගැනීම ආරම්භ කිරීමට T යතුර ඔබන්න"
                    
                    return JSONResponse({
                        "object": most_common_label,
                        "description": intro,
                        "detection_complete": True,
                        "audio": audio_b64,
                        "next_step": next_step,
                        "language": language,
                        "bounding_box": bounding_box
                    })
                else:
                    # Not confident enough
                    del rotation_sessions[session_id]
                    message = "Detection not confident. Please rotate the object again." if language == "en" else "හඳුනා ගැනීම ස්ථිර නැත. කරුණාකර වස්තුව නැවත කරකවන්න."
                    audio_b64 = speak_as_base64(message, language)
                    return JSONResponse({
                        "detection_complete": False,
                        "error": "Detection not confident enough",
                        "audio": audio_b64,
                        "language": language
                    })
            else:
                del rotation_sessions[session_id]
                message = "No object detected. Please try again." if language == "en" else "කිසිදු වස්තුවක් හඳුනා නොගැනිණි. කරුණාකර නැවත උත්සාහ කරන්න."
                audio_b64 = speak_as_base64(message, language)
                return JSONResponse({
                    "detection_complete": False,
                    "error": "No object detected",
                    "audio": audio_b64,
                    "language": language
                })
        
        # Still collecting frames
        return JSONResponse({
            "detection_complete": False,
            "frame_count": session["frame_count"],
            "target_frames": session["target_frames"],
            "current_detection": detected_label,
            "bounding_box": bounding_box,
            "progress": session["frame_count"] / session["target_frames"],
            "language": language
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
async def detect_feature(file: UploadFile = File(...), language: str = Form("en")):
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
            label_raw = touch_model.names[cls_id]
            label = label_raw.strip().lower()
            
            # Get translated feature name
            feature_name = feature_translations[language].get(label, label)
            
            # Get bounding box for visualization
            box = results.boxes[0]
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            
            return JSONResponse({
                "feature": feature_name,
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
    language = data.get("language", "en")
    is_next_instruction = data.get("is_next_instruction", False)
    
    try:
        if is_next_instruction:
            text = "Please move to the next feature" if language == "en" else "කරුණාකර ඊළඟ ලක්ෂණයට යන්න"
        else:
            # Get translated feature name if available
            feature_name = feature_translations[language].get(feature, feature)
            text = f"You touched a {feature_name}" if language == "en" else f"ඔබ {feature_name} ස්පර්ශ කළා"
        
        audio_b64 = speak_as_base64(text, language)
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
    language = data.get("language", "en")  # Default to English if not specified
    
    if not text:
        return JSONResponse(status_code=400, content={"error": "No text provided"})
    
    audio_b64 = speak_as_base64(text, language)
    return JSONResponse({"audio": audio_b64})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


