import React, { useEffect, useRef, useState, useCallback } from "react";
import axios from "axios";

function App() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [isStreaming, setIsStreaming] = useState(false);
  
  // State management
  const [mode, setMode] = useState("none"); // none, object, rotation, feature
  const [objectDetected, setObjectDetected] = useState(false);
  const [detectedObjectName, setDetectedObjectName] = useState("");
  const [readyForFeatures, setReadyForFeatures] = useState(false);
  const [featuresMode, setFeaturesMode] = useState(false);
  
  // Feature detection state
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

  // Play audio from base64
  const playAudio = useCallback((base64Audio) => {
    if (!base64Audio) return;
    const audio = new Audio(`data:audio/mp3;base64,${base64Audio}`);
    audio.play().catch(e => console.log("Audio play failed:", e));
  }, []);

  // Initialize webcam
  const startWebcam = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { width: 640, height: 480 } 
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        setIsStreaming(true);
        setStatus("Webcam started. Press 'F' to detect object.");
      }
    } catch (err) {
      setError("Failed to access webcam: " + err.message);
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

  // Start rotation detection
  const startRotationDetection = useCallback(async () => {
    setMode("rotation");
    setStatus("Please hold and slowly rotate the object for analysis...");
    
    try {
      // First play the instruction
      const instructionResponse = await axios.post(`${API_BASE_URL}/speak/`, {
        text: "Please hold and slowly rotate the object for analysis"
      });

      if (instructionResponse.data.audio) {
        const audio = new Audio(`data:audio/mp3;base64,${instructionResponse.data.audio}`);
        
        // Wait for the audio to finish before starting the analysis
        await new Promise((resolve) => {
          audio.onended = resolve;
          audio.play().catch(e => {
            console.log("Audio play failed:", e);
            resolve(); // Resolve anyway to prevent hanging
          });
        });

        // Now start the rotation detection
        setIsRotating(true);
        setRotationProgress(0);
        
        const response = await axios.post(`${API_BASE_URL}/start-rotation-detection/`, {
          session_id: Date.now().toString()
        });
        setRotationSession(response.data.session_id);
      }
    } catch (err) {
      setError("Failed to start rotation detection: " + err.message);
      setMode("none");
      setIsRotating(false);
    }
  }, []);

  // Process rotation frames
  const processRotationFrame = useCallback(async () => {
    if (mode !== "rotation" || !isRotating) return;
    
    try {
      const blob = await captureFrame();
      if (!blob) return;

      const formData = new FormData();
      formData.append("file", blob, "frame.jpg");

      const response = await axios.post(`${API_BASE_URL}/detect-object-rotation/`, formData);
      const data = response.data;

      // Draw bounding box if object is detected in current frame
      const ctx = canvasRef.current.getContext('2d');
      ctx.clearRect(0, 0, 640, 480);

      if (data.bounding_box) {
        const { x1, y1, x2, y2 } = data.bounding_box;
        
        // Draw rectangle
        ctx.strokeStyle = '#2196F3'; // Blue color for object detection
        ctx.lineWidth = 2;
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
        
        // Draw text
        if (data.current_detection) {
          ctx.fillStyle = '#2196F3';
          ctx.font = '20px Arial';
          ctx.fillText(data.current_detection.replace("_", " "), x1, y1 - 10);
        }
      }

      if (data.detection_complete) {
        if (data.object) {
          setDetectedObjectName(data.object);
          setObjectDetected(true);
          setReadyForFeatures(true);
          setStatus(`Detected: ${data.object.replace('_', ' ')}. Press 'T' to start feature detection.`);
          setMode("none");
          setIsRotating(false);

          // Clear canvas when detection is complete
          ctx.clearRect(0, 0, 640, 480);

          // Play the object detection result first
          if (data.audio) {
            const objectAudio = new Audio(`data:audio/mp3;base64,${data.audio}`);
            await new Promise((resolve) => {
              objectAudio.onended = resolve;
              objectAudio.play().catch(e => {
                console.log("Audio play failed:", e);
                resolve(); // Resolve anyway to prevent hanging
              });
            });

            // After object audio finishes, play the "Press T" instruction
            try {
              const featureResponse = await axios.post(`${API_BASE_URL}/speak/`, {
                text: " Lets go to the Feature Detection. Before we begin, please wear a glove on the hand that holds the shape. only Use your index finger of bare hand to touch the object's features.don't use other fingers to touch. This helps the system recognize your touch correctly.Press T to start feature detection."
              });
              if (featureResponse.data.audio) {
                playAudio(featureResponse.data.audio);
              }
            } catch (err) {
              console.log("Failed to play feature instruction:", err.message);
            }
          }
        } else {
          setStatus("Detection failed. Please try again.");
          if (data.audio) {
            playAudio(data.audio);
          }
          setMode("none");
          setIsRotating(false);
          // Clear canvas on failure
          ctx.clearRect(0, 0, 640, 480);
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
        // Clear canvas on error
        const ctx = canvasRef.current.getContext('2d');
        ctx.clearRect(0, 0, 640, 480);
      }
    }
  }, [mode, isRotating, captureFrame, playAudio]);

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

      // Draw bounding box if feature is detected
      if (data.feature && data.bounding_box) {
        const ctx = canvasRef.current.getContext('2d');
        const { x1, y1, x2, y2 } = data.bounding_box;
        
        // Clear previous drawing
        ctx.clearRect(0, 0, 640, 480);
        
        // Draw rectangle
        ctx.strokeStyle = '#00FF00';
        ctx.lineWidth = 2;
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
        
        // Draw text
        ctx.fillStyle = '#00FF00';
        ctx.font = '20px Arial';
        ctx.fillText(data.feature.replace("_", " "), x1, y1 - 10);
      } else {
        // Clear canvas if no feature detected
        const ctx = canvasRef.current.getContext('2d');
        ctx.clearRect(0, 0, 640, 480);
      }

      // If a feature is being processed, don't do anything else
      if (data.is_processing) {
        return;
      }

      const foundFeature = data.feature;
      const now = Date.now();

      if (foundFeature) {
        if (foundFeature !== currentFeature && now > featureCooldownUntil) {
          // New feature detected and not in cooldown
          setCurrentFeature(foundFeature);
          setFeatureHoldStart(now);
        } else if (foundFeature === currentFeature && 
                   featureHoldStart && 
                   (now - featureHoldStart >= FEATURE_HOLD_SECONDS)) {
        
        // Start the feature announcement process
        await axios.post(`${API_BASE_URL}/start-feature-announcement/`);

        try {
          // First announce the detected feature
          const featureResponse = await axios.post(`${API_BASE_URL}/speak-feature/`, {
            feature: foundFeature,
            is_next_instruction: false
          });
          
          if (featureResponse.data.audio) {
            const audio = new Audio(`data:audio/mp3;base64,${featureResponse.data.audio}`);
            await new Promise((resolve) => {
              audio.onended = resolve;
              audio.play().catch(() => resolve());
            });
          }

          // Then tell to move to next feature
          const nextResponse = await axios.post(`${API_BASE_URL}/speak-feature/`, {
            is_next_instruction: true
          });
          
          if (nextResponse.data.audio) {
            const audio = new Audio(`data:audio/mp3;base64,${nextResponse.data.audio}`);
            await new Promise((resolve) => {
              audio.onended = resolve;
              audio.play().catch(() => resolve());
            });
          }

          // After both audio messages complete, reset for next feature
          setFeatureCooldownUntil(Date.now() + FEATURE_COOLDOWN_SECONDS);
          setCurrentFeature(null);
          setFeatureHoldStart(null);
        } finally {
          // Always end the announcement process, even if there was an error
          await axios.post(`${API_BASE_URL}/end-feature-announcement/`);
        }
      }
    } else if (now > featureCooldownUntil) {
      // Only reset if we're not in cooldown
      setCurrentFeature(null);
      setFeatureHoldStart(null);
    }
  } catch (err) {
    console.log("Feature detection error:", err);
    // Clear canvas on error
    const ctx = canvasRef.current.getContext('2d');
    ctx.clearRect(0, 0, 640, 480);
    
    if (Date.now() > featureCooldownUntil) {
      setCurrentFeature(null);
      setFeatureHoldStart(null);
    }
  }
}, [mode, featuresMode, currentFeature, featureHoldStart, featureCooldownUntil, captureFrame]);

  // Start feature detection mode
  const startFeatureDetection = useCallback(() => {
    if (!objectDetected || !readyForFeatures) {
      setStatus("Please detect an object first!");
      return;
    }
    
    setMode("feature");
    setFeaturesMode(true);
    setFeatureCooldownUntil(0);
    setCurrentFeature(null);
    setFeatureHoldStart(null);
    setStatus("Feature detection started. Touch a feature and hold your finger on it.");
    
    // Play initial feature detection instruction
    axios.post(`${API_BASE_URL}/speak/`, {
      text: "Feature detection started. Touch a feature and hold your finger on it."
    }).then(response => {
      if (response.data.audio) {
        const audio = new Audio(`data:audio/mp3;base64,${response.data.audio}`);
        audio.play().catch(e => console.log("Audio play failed:", e));
      }
    }).catch(err => {
      console.log("Failed to play feature detection instruction:", err.message);
    });
  }, [objectDetected, readyForFeatures]);

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

  // Set up event listeners and intervals
  useEffect(() => {
    window.addEventListener("keydown", handleKeyPress);
    return () => window.removeEventListener("keydown", handleKeyPress);
  }, [handleKeyPress]);

  useEffect(() => {
    if (mode === "rotation" && isRotating) {
      const interval = setInterval(processRotationFrame, 200);
      return () => clearInterval(interval);
    }
  }, [mode, isRotating, processRotationFrame]);

  useEffect(() => {
    if (mode === "feature" && featuresMode) {
      const interval = setInterval(processFeatureDetection, 500);
      return () => clearInterval(interval);
    }
  }, [mode, featuresMode, processFeatureDetection]);

  // Start webcam when app loads
  useEffect(() => {
    startWebcam();
  }, [startWebcam]);

  // Play welcome messages after component is mounted and webcam is ready
  useEffect(() => {
    const playWelcomeMessages = async () => {
      if (!isStreaming) return; // Only play when webcam is ready
      
      try {
        // First play welcome message
        const welcomeResponse = await axios.post(`${API_BASE_URL}/speak/`, {
          text: "Welcome to the object detection learning platform"
        });
        
        if (welcomeResponse.data.audio) {
          const welcomeAudio = new Audio(`data:audio/mp3;base64,${welcomeResponse.data.audio}`);
          await new Promise((resolve) => {
            welcomeAudio.onended = resolve;
            welcomeAudio.play().catch(() => resolve());
          });
          
          // After welcome message finishes, play instruction
          const instructionResponse = await axios.post(`${API_BASE_URL}/speak/`, {
            text: "Press F to start object detection"
          });
          
          if (instructionResponse.data.audio) {
            const instructionAudio = new Audio(`data:audio/mp3;base64,${instructionResponse.data.audio}`);
            instructionAudio.play().catch(e => console.log("Audio play failed:", e));
          }
        }
      } catch (err) {
        console.log("Failed to play audio messages:", err.message);
      }
    };

    playWelcomeMessages();
  }, [isStreaming]); // Only run when isStreaming changes to true

  return (
    <div style={{ textAlign: "center", padding: "2rem", fontFamily: "Arial, sans-serif" }}>
      <h2>🧠 3D Object Detection & Feature Teaching</h2>
      
      {/* Video and Canvas */}
      <div style={{ margin: "1rem 0", position: "relative" }}>
        <video 
          ref={videoRef} 
          width="640" 
          height="480" 
          autoPlay 
          muted 
          style={{ 
            border: "2px solid #333", 
            borderRadius: "8px",
            display: "block" // Ensure video is block-level
          }}
        />
        <canvas 
          ref={canvasRef} 
          width="640" 
          height="480" 
          style={{
            position: "absolute",
            left: 0,
            top: 0,
            pointerEvents: "none",
            borderRadius: "8px"
          }}
        />
      </div>

      {/* Progress Bar */}
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
        margin: "2rem auto", 
        padding: "1rem", 
        backgroundColor: "#0f0707ff", 
        borderRadius: "5px",
        textAlign: "left",
        maxWidth: "640px"
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
