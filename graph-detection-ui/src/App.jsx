import React, { useEffect, useRef, useState, useCallback } from "react";
import axios from "axios";

function App() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [isStreaming, setIsStreaming] = useState(false);
  
  // Audio queue management
  const [audioQueue, setAudioQueue] = useState([]);
  const [isPlaying, setIsPlaying] = useState(false);
  const currentAudioRef = useRef(null);
  
  // State management
  const [mode, setMode] = useState("none"); // none, object, rotation, feature
  const [objectDetected, setObjectDetected] = useState(false);
  const [detectedObjectName, setDetectedObjectName] = useState("");
  const [readyForFeatures, setReadyForFeatures] = useState(false);
  const [featuresMode, setFeaturesMode] = useState(false);
  
  // Feature detection state (matching original code)
  const [currentFeature, setCurrentFeature] = useState(null);
  const [featureHoldStart, setFeatureHoldStart] = useState(null);
  const [featureCooldownUntil, setFeatureCooldownUntil] = useState(0);
  const FEATURE_HOLD_SECONDS = 2000; // 2 seconds in milliseconds
  const FEATURE_COOLDOWN_SECONDS = 10000; // 10 seconds in milliseconds
  
  // Rotation detection state
  const [rotationSession, setRotationSession] = useState(null);
  const [rotationProgress, setRotationProgress] = useState(0);
  const [isRotating, setIsRotating] = useState(false);
  
  // Status and feedback
  const [status, setStatus] = useState("Press 'F' to start object detection");
  const [error, setError] = useState("");

  const API_BASE_URL = "http://localhost:8000";

  // Initialize webcam and play initial instruction
  const startWebcam = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { width: 640, height: 480 } 
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        setIsStreaming(true);
        setStatus("Webcam started. Press 'F' to detect object.");
        
        // Play initial instruction
        setTimeout(() => {
          playInitialInstruction();
        }, 1000);
      }
    } catch (err) {
      setError("Failed to access webcam: " + err.message);
    }
  }, []);

  // Play initial instruction when app loads
  const playInitialInstruction = useCallback(async () => {
    try {
      const response = await axios.post(`${API_BASE_URL}/speak/`, {
        text: "Press F to start object detection"
      });
      playAudio(response.data.audio);
    } catch (err) {
      console.log("Failed to play initial instruction:", err.message);
    }
  }, []);

  // Capture frame from video
  const captureFrame = useCallback(async () => {
    if (!videoRef.current || !canvasRef.current) return null;
    
    const context = canvasRef.current.getContext("2d");
    context.drawImage(videoRef.current, 0, 0, 640, 480);
    
    return new Promise((resolve) => {
      canvasRef.current.toBlob(resolve, "image/jpeg", 0.8);
    });
  }, []);

  // Queue audio for playback
  const queueAudio = useCallback((base64Audio) => {
    if (!base64Audio) return;
    setAudioQueue(prev => [...prev, base64Audio]);
  }, []);

  // Play audio from base64
  const playAudio = useCallback((base64Audio) => {
    queueAudio(base64Audio);
  }, [queueAudio]);

  // Process audio queue
  useEffect(() => {
    const playNext = () => {
      if (audioQueue.length > 0 && !isPlaying) {
        setIsPlaying(true);
        const audio = new Audio(`data:audio/mp3;base64,${audioQueue[0]}`);
        currentAudioRef.current = audio;
        
        audio.onended = () => {
          setAudioQueue(prev => prev.slice(1));
          setIsPlaying(false);
          currentAudioRef.current = null;
        };

        audio.onerror = () => {
          console.log("Audio play failed");
          setAudioQueue(prev => prev.slice(1));
          setIsPlaying(false);
          currentAudioRef.current = null;
        };

        audio.play().catch(e => {
          console.log("Audio play failed:", e);
          setAudioQueue(prev => prev.slice(1));
          setIsPlaying(false);
          currentAudioRef.current = null;
        });
      }
    };

    playNext();
  }, [audioQueue, isPlaying]);

  // Clean up audio on unmount or reset
  useEffect(() => {
    return () => {
      if (currentAudioRef.current) {
        currentAudioRef.current.pause();
        currentAudioRef.current = null;
      }
      setAudioQueue([]);
      setIsPlaying(false);
    };
  }, []);

  // Start rotation-based object detection
  const startRotationDetection = useCallback(async () => {
    setMode("rotation");
    setIsRotating(true);
    setRotationProgress(0);
    setStatus("Please hold and slowly rotate the object for analysis...");
    
    try {
      const response = await axios.post(`${API_BASE_URL}/start-rotation-detection/`, {
        session_id: Date.now().toString()
      });
      setRotationSession(response.data.session_id);
      playAudio(response.data.audio);
    } catch (err) {
      setError("Failed to start rotation detection: " + err.message);
      setMode("none");
      setIsRotating(false);
    }
  }, [playAudio]);

  // Process rotation detection frames
  const processRotationFrame = useCallback(async () => {
    if (mode !== "rotation" || !isRotating) return;
    
    try {
      const blob = await captureFrame();
      if (!blob) return;

      const formData = new FormData();
      formData.append("file", blob, "frame.jpg");

      const response = await axios.post(`${API_BASE_URL}/detect-object-rotation/`, formData);
      const data = response.data;

      if (data.detection_complete) {
        if (data.object) {
          setDetectedObjectName(data.object);
          setObjectDetected(true);
          setReadyForFeatures(true);
          setStatus(`Detected: ${data.object.replace('_', ' ')}. Press 'T' to start feature detection.`);
          playAudio(data.audio);
          setMode("none");
          setIsRotating(false);
          
          // Play feature detection instructions after object detection
          setTimeout(async () => {
            try {
              const featureResponse = await axios.post(`${API_BASE_URL}/speak/`, {
                text: "Press T to start feature detection"
              });
              playAudio(featureResponse.data.audio);
              
              // Play guidelines after a short delay
              setTimeout(async () => {
                try {
                  const guidelineResponse = await axios.post(`${API_BASE_URL}/speak/`, {
                    text: "Before we begin, please wear a glove on the hand that holds the shape. Only use your index finger of bare hand to touch the object's features. Don't use other fingers to touch. This helps the system recognize your touch correctly."
                  });
                  playAudio(guidelineResponse.data.audio);
                } catch (err) {
                  console.log("Failed to play guidelines:", err.message);
                }
              }, 3000);
            } catch (err) {
              console.log("Failed to play feature instruction:", err.message);
            }
          }, 2000);
        } else {
          setStatus("Detection failed. Please try again.");
          playAudio(data.audio);
          setMode("none");
          setIsRotating(false);
        }
      } else {
        setRotationProgress(data.progress || 0);
        setStatus(`Analyzing... ${Math.round((data.progress || 0) * 100)}%`);
      }
    } catch (err) {
      console.log("Rotation detection error:", err.message);
      if (err.response?.status !== 400) {
        setError("Rotation detection failed: " + err.message);
        setMode("none");
        setIsRotating(false);
      }
    }
  }, [mode, isRotating, captureFrame, playAudio]);

  // Simple object detection (fallback)
  const detectObject = useCallback(async () => {
    setMode("object");
    setStatus("Detecting object...");
    
    try {
      const blob = await captureFrame();
      if (!blob) return;

      const formData = new FormData();
      formData.append("file", blob, "frame.jpg");

      const response = await axios.post(`${API_BASE_URL}/detect-object/`, formData);
      const data = response.data;

      setDetectedObjectName(data.object);
      setObjectDetected(true);
      setReadyForFeatures(true);
      setStatus(`Detected: ${data.object.replace('_', ' ')}. Press 'T' to start feature detection.`);
      playAudio(data.audio);
      setMode("none");
      
      // Play feature detection instructions for simple detection too
      setTimeout(async () => {
        try {
          const featureResponse = await axios.post(`${API_BASE_URL}/speak/`, {
            text: "Press T to start feature detection"
          });
          playAudio(featureResponse.data.audio);
          
          setTimeout(async () => {
            try {
              const guidelineResponse = await axios.post(`${API_BASE_URL}/speak/`, {
                text: "Before we begin, please wear a glove on the hand that holds the shape. Only use your index finger of bare hand to touch the object's features. Don't use other fingers to touch. This helps the system recognize your touch correctly."
              });
              playAudio(guidelineResponse.data.audio);
            } catch (err) {
              console.log("Failed to play guidelines:", err.message);
            }
          }, 3000);
        } catch (err) {
          console.log("Failed to play feature instruction:", err.message);
        }
      }, 2000);
    } catch (err) {
      setError("Object detection failed: " + err.response?.data?.error || err.message);
      setMode("none");
    }
  }, [captureFrame, playAudio]);

  // Start feature detection mode
  const startFeatureDetection = useCallback(() => {
    if (!objectDetected || !readyForFeatures) {
      setStatus("Please detect an object first!");
      return;
    }
    
    setMode("feature");
    setFeaturesMode(true);
    setStatus("Feature detection started. Touch a feature and hold your finger on it.");
    playAudio("Feature detection started. Touch a feature and hold your finger on it.");
  }, [objectDetected, readyForFeatures, playAudio]);

  // Process feature detection
  const processFeatureDetection = useCallback(async () => {
    if (mode !== "feature" || !featuresMode) return;

    try {
      const blob = await captureFrame();
      if (!blob) return;

      const formData = new FormData();
      formData.append("file", blob, "frame.jpg");

      const response = await axios.post(`${API_BASE_URL}/detect-feature/`, formData);
      const data = response.data;

      const foundFeature = data.feature;
      const now = Date.now();

      if (foundFeature) {
        if (foundFeature !== currentFeature) {
          setCurrentFeature(foundFeature);
          setFeatureHoldStart(now);
        } else {
          if (featureHoldStart && 
              (now - featureHoldStart >= FEATURE_HOLD_SECONDS) && 
              (now > featureCooldownUntil)) {
            
            playAudio(data.audio);
            setFeatureCooldownUntil(now + FEATURE_COOLDOWN_SECONDS);
            setCurrentFeature(null);
            setFeatureHoldStart(null);
          }
        }
      } else {
        setCurrentFeature(null);
        setFeatureHoldStart(null);
      }
    } catch (err) {
      // Silently handle no feature detection
      setCurrentFeature(null);
      setFeatureHoldStart(null);
    }
  }, [mode, featuresMode, currentFeature, featureHoldStart, featureCooldownUntil, captureFrame, playAudio]);

  // Reset all states
  const resetApp = useCallback(() => {
    setMode("none");
    setObjectDetected(false);
    setDetectedObjectName("");
    setReadyForFeatures(false);
    setFeaturesMode(false);
    setCurrentFeature(null);
    setFeatureHoldStart(null);
    setFeatureCooldownUntil(0);
    setIsRotating(false);
    setRotationProgress(0);
    setRotationSession(null);
    setStatus("Press 'F' to start object detection");
    setError("");
  }, []);

  // Keyboard event handler
  const handleKeyPress = useCallback((e) => {
    const key = e.key.toLowerCase();
    
    if (key === 'f') {
      e.preventDefault();
      if (!isStreaming) {
        setStatus("Please start webcam first!");
        return;
      }
      resetApp();
      startRotationDetection();
    } else if (key === 't') {
      e.preventDefault();
      startFeatureDetection();
    } else if (key === 'q') {
      e.preventDefault();
      resetApp();
    }
  }, [isStreaming, resetApp, startRotationDetection, startFeatureDetection]);

  // Set up intervals and event listeners
  useEffect(() => {
    window.addEventListener("keydown", handleKeyPress);
    return () => window.removeEventListener("keydown", handleKeyPress);
  }, [handleKeyPress]);

  // Rotation detection interval
  useEffect(() => {
    if (mode === "rotation" && isRotating) {
      const interval = setInterval(processRotationFrame, 200); // Process every 200ms
      return () => clearInterval(interval);
    }
  }, [mode, isRotating, processRotationFrame]);

  // Feature detection interval
  useEffect(() => {
    if (mode === "feature" && featuresMode) {
      const interval = setInterval(processFeatureDetection, 500); // Process every 500ms
      return () => clearInterval(interval);
    }
  }, [mode, featuresMode, processFeatureDetection]);

  useEffect(() => {
    startWebcam();
  }, [startWebcam]);

  return (
    <div style={{ textAlign: "center", padding: "2rem", fontFamily: "Arial, sans-serif" }}>
      <h2>🧠 3D Object Detection & Feature Teaching</h2>
      
      {/* Video and Canvas */}
      <div style={{ margin: "1rem 0" }}>
        <video 
          ref={videoRef} 
          width="640" 
          height="480" 
          autoPlay 
          muted 
          style={{ border: "2px solid #333", borderRadius: "8px" }}
        />
        <canvas 
          ref={canvasRef} 
          width="640" 
          height="480" 
          hidden 
        />
      </div>

      {/* Progress Bar for Rotation Detection */}
      {isRotating && (
        <div style={{ margin: "1rem 0" }}>
          <div style={{ 
            width: "640px", 
            height: "20px", 
            backgroundColor: "#f0f0f0", 
            borderRadius: "10px",
            margin: "0 auto"
          }}>
            <div style={{
              width: `${rotationProgress * 100}%`,
              height: "100%",
              backgroundColor: "#4CAF50",
              borderRadius: "10px",
              transition: "width 0.3s ease"
            }}></div>
          </div>
          <p>Analysis Progress: {Math.round(rotationProgress * 100)}%</p>
        </div>
      )}

      {/* Control Buttons */}
      <div style={{ margin: "1rem 0" }}>
        <button 
          onClick={startWebcam}
          disabled={isStreaming}
          style={{ 
            margin: "0 0.5rem", 
            padding: "0.5rem 1rem",
            fontSize: "1rem",
            backgroundColor: isStreaming ? "#ccc" : "#4CAF50",
            color: "white",
            border: "none",
            borderRadius: "5px",
            cursor: isStreaming ? "not-allowed" : "pointer"
          }}
        >
          {isStreaming ? "Webcam Active" : "Start Webcam"}
        </button>
        
        <button 
          onClick={startRotationDetection}
          disabled={!isStreaming || isRotating}
          style={{ 
            margin: "0 0.5rem", 
            padding: "0.5rem 1rem",
            fontSize: "1rem",
            backgroundColor: (!isStreaming || isRotating) ? "#ccc" : "#2196F3",
            color: "white",
            border: "none",
            borderRadius: "5px",
            cursor: (!isStreaming || isRotating) ? "not-allowed" : "pointer"
          }}
        >
          Detect Object (F)
        </button>
        
        <button 
          onClick={startFeatureDetection}
          disabled={!objectDetected || !readyForFeatures}
          style={{ 
            margin: "0 0.5rem", 
            padding: "0.5rem 1rem",
            fontSize: "1rem",
            backgroundColor: (!objectDetected || !readyForFeatures) ? "#ccc" : "#FF9800",
            color: "white",
            border: "none",
            borderRadius: "5px",
            cursor: (!objectDetected || !readyForFeatures) ? "not-allowed" : "pointer"
          }}
        >
          Start Features (T)
        </button>
        
        <button 
          onClick={resetApp}
          style={{ 
            margin: "0 0.5rem", 
            padding: "0.5rem 1rem",
            fontSize: "1rem",
            backgroundColor: "#f44336",
            color: "white",
            border: "none",
            borderRadius: "5px",
            cursor: "pointer"
          }}
        >
          Reset (Q)
        </button>
      </div>

      {/* Status Display */}
      <div style={{ margin: "1rem 0" }}>
        <h3>Status:</h3>
        <p style={{ 
          padding: "1rem", 
          backgroundColor: "#121b22ff", 
          borderRadius: "5px",
          border: "1px solid #ddd",
          fontWeight: "bold"
        }}>
          {status}
        </p>
        
        {detectedObjectName && (
          <p style={{ color: "#4CAF50", fontWeight: "bold" }}>
            Detected Object: {detectedObjectName.replace('_', ' ')}
          </p>
        )}
        
        {currentFeature && (
          <p style={{ color: "#FF9800", fontWeight: "bold" }}>
            Current Feature: {currentFeature}
          </p>
        )}
      </div>

      {/* Error Display */}
      {error && (
        <div style={{ 
          margin: "1rem 0", 
          padding: "1rem", 
          backgroundColor: "#100d0eff", 
          color: "#c62828",
          borderRadius: "5px",
          border: "1px solid #f44336"
        }}>
          <strong>Error:</strong> {error}
          <button 
            onClick={() => setError("")}
            style={{ 
              marginLeft: "1rem", 
              padding: "0.25rem 0.5rem",
              backgroundColor: "#f44336",
              color: "white",
              border: "none",
              borderRadius: "3px",
              cursor: "pointer"
            }}
          >
            Clear
          </button>
        </div>
      )}

      {/* Instructions */}
      <div style={{ 
        margin: "2rem 0", 
        padding: "1rem", 
        backgroundColor: "#0f0707ff", 
        borderRadius: "5px",
        textAlign: "left",
        maxWidth: "640px",
        margin: "2rem auto"
      }}>
        <h4>Instructions:</h4>
        <ul style={{ paddingLeft: "1.5rem" }}>
          <li><strong>F Key:</strong> Start object detection with rotation analysis</li>
          <li><strong>T Key:</strong> Begin feature detection (after object is detected)</li>
          <li><strong>Q Key:</strong> Reset and quit current mode</li>
          <li><strong>Feature Detection:</strong> Wear a glove on the hand holding the object</li>
          <li><strong>Feature Detection:</strong> Use only your index finger to touch features</li>
          <li><strong>Hold Time:</strong> Hold finger on feature for 2 seconds to register</li>
        </ul>
      </div>
    </div>
  );
}

export default App;
