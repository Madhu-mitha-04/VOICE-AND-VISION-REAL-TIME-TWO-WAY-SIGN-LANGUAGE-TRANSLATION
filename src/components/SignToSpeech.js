import React, { useState, useRef, useEffect, useCallback } from 'react';
import Webcam from 'react-webcam';
import { 
  Container, 
  Typography, 
  Button, 
  Box, 
  Radio, 
  RadioGroup, 
  FormControlLabel, 
  FormControl,
  CircularProgress,
  Paper,
  IconButton,
  Tooltip
} from '@mui/material';
import VolumeUpIcon from '@mui/icons-material/VolumeUp';
import VolumeOffIcon from '@mui/icons-material/VolumeOff';
import ReplayIcon from '@mui/icons-material/Replay';
import InfoIcon from '@mui/icons-material/Info';

const WS_URL = 'ws://localhost:5000/realtime';

const SignToSpeech = () => {
  // State management
  const [inputType, setInputType] = useState('webcam');
  const [prediction, setPrediction] = useState('');
  const [confidence, setConfidence] = useState(0);
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState('');
  const [videoFile, setVideoFile] = useState(null);
  const [isWebcamActive, setIsWebcamActive] = useState(false);
  const [bufferStatus, setBufferStatus] = useState(0);
  const [isMuted, setIsMuted] = useState(false);
  const [lastSpokenPrediction, setLastSpokenPrediction] = useState('');
  const [isConnected, setIsConnected] = useState(false);
  const [connectionStatus, setConnectionStatus] = useState('disconnected');
  const [showHelp, setShowHelp] = useState(false);

  // Refs
  const webcamRef = useRef(null);
  const videoRef = useRef(null);
  const ws = useRef(null);
  const frameInterval = useRef(null);
  const reconnectAttempt = useRef(0);
  const maxReconnectAttempts = 5;
  const reconnectDelay = 3000; // 3 seconds

  const videoConstraints = {
    width: 480,
    height: 360,
    facingMode: "user"
  };

  // WebSocket connection management
  const connectWebSocket = useCallback(() => {
    if (ws.current && (ws.current.readyState === WebSocket.OPEN || ws.current.readyState === WebSocket.CONNECTING)) {
      return;
    }

    setConnectionStatus('connecting');
    ws.current = new WebSocket(WS_URL);

    ws.current.onopen = () => {
      console.log('WebSocket connected');
      setIsConnected(true);
      setConnectionStatus('connected');
      reconnectAttempt.current = 0;
    };

    ws.current.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        
        if (data.type === 'prediction') {
          setPrediction(data.prediction);
          setConfidence(data.confidence);
          setIsProcessing(false);
        } 
        else if (data.type === 'buffer_status') {
          setBufferStatus(data.status);
        }
        else if (data.type === 'error') {
          setError(data.message);
          setIsProcessing(false);
        }
        else if (data.type === 'pong') {
          // Heartbeat response
        }
      } catch (err) {
        console.error('Error parsing WebSocket message:', err);
      }
    };

    ws.current.onclose = () => {
      console.log('WebSocket disconnected');
      setIsConnected(false);
      setConnectionStatus('disconnected');
      
      // Attempt reconnection if we were connected before
      if (isWebcamActive && reconnectAttempt.current < maxReconnectAttempts) {
        reconnectAttempt.current += 1;
        const delay = Math.min(reconnectDelay, reconnectAttempt.current * 1000);
        setTimeout(connectWebSocket, delay);
        setConnectionStatus(`reconnecting (${reconnectAttempt.current}/${maxReconnectAttempts})`);
      }
    };

    ws.current.onerror = (error) => {
      console.error('WebSocket error:', error);
      setError('Connection error');
      setConnectionStatus('error');
    };
  }, [isWebcamActive]);

  // Initialize WebSocket connection
  useEffect(() => {
    connectWebSocket();
    
    return () => {
      if (ws.current) {
        ws.current.close();
      }
      if (frameInterval.current) {
        clearInterval(frameInterval.current);
      }
    };
  }, [connectWebSocket]);

  // Heartbeat to keep connection alive
  useEffect(() => {
    const heartbeatInterval = setInterval(() => {
      if (ws.current && ws.current.readyState === WebSocket.OPEN) {
        ws.current.send(JSON.stringify({ type: 'ping' }));
      }
    }, 25000); // 25 seconds

    return () => clearInterval(heartbeatInterval);
  }, []);

  // Text-to-speech functionality
  const speakPrediction = useCallback((text) => {
    if ('speechSynthesis' in window && !isMuted && text) {
      window.speechSynthesis.cancel();
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.rate = 0.9;
      utterance.pitch = 1;
      utterance.volume = 1;
      window.speechSynthesis.speak(utterance);
    }
  }, [isMuted]);

  // Speak the prediction when it changes
  useEffect(() => {
    if (prediction && prediction !== lastSpokenPrediction) {
      speakPrediction(prediction);
      setLastSpokenPrediction(prediction);
    }
  }, [prediction, lastSpokenPrediction, speakPrediction]);

  const toggleMute = () => {
    const newMutedState = !isMuted;
    setIsMuted(newMutedState);
    if (!newMutedState && prediction) {
      speakPrediction(prediction);
    } else {
      window.speechSynthesis?.cancel();
    }
  };

  // Handle input type change
  const handleInputTypeChange = (type) => {
    setInputType(type);
    setPrediction('');
    setConfidence(0);
    setError('');
    setIsProcessing(false);
    setBufferStatus(0);
    
    if (frameInterval.current) {
      clearInterval(frameInterval.current);
      frameInterval.current = null;
    }

    if (type === 'webcam') {
      setVideoFile(null);
    } else {
      setIsWebcamActive(false);
    }
  };

  // Start webcam predictions
  const startWebcamPredictions = () => {
    if (!ws.current || ws.current.readyState !== WebSocket.OPEN) {
      setError('Not connected to server');
      connectWebSocket();
      return;
    }

    setIsWebcamActive(true);
    setError('');
    setBufferStatus(0);
    setPrediction('');
    setConfidence(0);
    
    // Clear any existing interval
    if (frameInterval.current) {
      clearInterval(frameInterval.current);
    }

    // Start new frame capture interval (~6-7fps)
    frameInterval.current = setInterval(() => {
      if (webcamRef.current && ws.current.readyState === WebSocket.OPEN) {
        const imageSrc = webcamRef.current.getScreenshot();
        if (imageSrc) {
          setIsProcessing(true);
          ws.current.send(JSON.stringify({
            type: 'frame',
            frame: imageSrc,
            timestamp: Date.now()
          }));
        }
      }
    }, 150);
  };

  // Stop webcam predictions
  const stopWebcamPredictions = () => {
    setIsWebcamActive(false);
    if (frameInterval.current) {
      clearInterval(frameInterval.current);
      frameInterval.current = null;
    }
    setPrediction('');
    setConfidence(0);
    setBufferStatus(0);
    setIsProcessing(false);
  };

  // Handle video file selection
  const handleVideoFileChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      setVideoFile(file);
      setPrediction('');
      setConfidence(0);
      setError('');
      setBufferStatus(0);
    }
  };

  // Process video file (using traditional REST API)
  const processVideo = async () => {
    if (!videoFile) {
      setError('Please select a video file');
      return;
    }

    setIsProcessing(true);
    setError('');

    try {
      const formData = new FormData();
      formData.append('video', videoFile);

      const response = await fetch('http://localhost:5000/predict_video', {
        method: 'POST',
        body: formData
      });

      const data = await response.json();
      
      if (data.status === 'success') {
        setPrediction(data.prediction);
        setConfidence(data.confidence);
      } else {
        setError(data.error || 'Failed to process video');
      }
    } catch (err) {
      console.error('Error processing video:', err);
      setError('Failed to process video');
    } finally {
      setIsProcessing(false);
    }
  };

  // Connection status indicator
  const getConnectionStatusColor = () => {
    switch (connectionStatus) {
      case 'connected': return 'success.main';
      case 'connecting': return 'warning.main';
      case 'reconnecting': return 'warning.main';
      case 'error': return 'error.main';
      default: return 'error.main';
    }
  };

  const getConnectionStatusText = () => {
    switch (connectionStatus) {
      case 'connected': return 'Connected';
      case 'connecting': return 'Connecting...';
      case 'reconnecting': return 'Reconnecting...';
      case 'error': return 'Connection Error';
      default: return 'Disconnected';
    }
  };

  return (
    <Container maxWidth="md" sx={{ py: 4 }}>
      <Paper elevation={3} sx={{ p: 4, borderRadius: 2 }}>
        {/* Header with connection status and controls */}
        <Box sx={{ 
          display: 'flex', 
          justifyContent: 'space-between', 
          alignItems: 'center', 
          mb: 2,
          flexWrap: 'wrap',
          gap: 2
        }}>
          <Typography
            variant="h4"
            component="h1"
            sx={{ 
              fontWeight: 'bold',
              color: 'primary.main',
            }}
          >
            Sign Language Recognition
          </Typography>
          
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
            {/* Connection status */}
            <Box sx={{ 
              display: 'flex', 
              alignItems: 'center',
              color: getConnectionStatusColor()
            }}>
              <Box sx={{
                width: 10,
                height: 10,
                borderRadius: '50%',
                bgcolor: getConnectionStatusColor(),
                mr: 1
              }} />
              <Typography variant="caption">
                {getConnectionStatusText()}
              </Typography>
            </Box>
            
            {/* Help button */}
            <Tooltip title="Help">
              <IconButton 
                onClick={() => setShowHelp(!showHelp)}
                color={showHelp ? 'primary' : 'default'}
              >
                <InfoIcon />
              </IconButton>
            </Tooltip>
            
            {/* Mute toggle */}
            <Tooltip title={isMuted ? "Unmute" : "Mute"}>
              <IconButton onClick={toggleMute} color="primary">
                {isMuted ? <VolumeOffIcon /> : <VolumeUpIcon />}
              </IconButton>
            </Tooltip>
          </Box>
        </Box>

        {/* Help section */}
        {showHelp && (
          <Paper elevation={1} sx={{ p: 2, mb: 3, backgroundColor: 'background.paper' }}>
            <Typography variant="body2" sx={{ mb: 1 }}>
              <strong>Real-time Webcam Mode:</strong> 
              <ul>
                <li>Click "Start Recognition" to begin</li>
                <li>Perform sign language gestures in front of your camera</li>
                <li>The system will analyze your gestures in real-time</li>
                <li>Click "Stop Recognition" to pause</li>
              </ul>
            </Typography>
            <Typography variant="body2">
              <strong>Video Upload Mode:</strong> 
              <ul>
                <li>Upload a video file containing sign language gestures</li>
                <li>Click "Process Video" to analyze</li>
                <li>The system will process the video and show results</li>
              </ul>
            </Typography>
          </Paper>
        )}

        {/* Input type selection */}
        <FormControl component="fieldset" sx={{ mb: 3, width: '100%' }}>
          <RadioGroup
            row
            aria-label="input-type"
            name="input-type"
            value={inputType}
            onChange={(event) => handleInputTypeChange(event.target.value)}
            sx={{ justifyContent: 'center' }}
          >
            <FormControlLabel 
              value="webcam" 
              control={<Radio />} 
              label="Real-time Webcam" 
            />
            <FormControlLabel 
              value="video" 
              control={<Radio />} 
              label="Upload Video" 
            />
          </RadioGroup>
        </FormControl>

        {/* Video display area */}
        <Box sx={{ 
          display: 'flex', 
          flexDirection: 'column', 
          alignItems: 'center',
          mb: 3
        }}>
          {inputType === 'webcam' ? (
            <Box sx={{
              width: '100%',
              maxWidth: '480px',
              position: 'relative',
              border: '2px solid #ddd',
              borderRadius: 1,
              overflow: 'hidden'
            }}>
              <Webcam
                audio={false}
                ref={webcamRef}
                width={480}
                height={360}
                screenshotFormat="image/jpeg"
                videoConstraints={videoConstraints}
                style={{
                  display: 'block',
                  width: '100%',
                  height: 'auto'
                }}
                onUserMedia={() => setIsWebcamActive(true)}
                onUserMediaError={() => {
                  setIsWebcamActive(false);
                  setError('Could not access webcam. Please check permissions.');
                }}
              />
              {!isWebcamActive && (
                <Box sx={{
                  position: 'absolute',
                  top: 0,
                  left: 0,
                  right: 0,
                  bottom: 0,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  backgroundColor: 'rgba(0,0,0,0.5)',
                  color: 'white'
                }}>
                  <Typography>Camera not available</Typography>
                </Box>
              )}
              {bufferStatus > 0 && bufferStatus < 100 && (
                <Box sx={{
                  position: 'absolute',
                  bottom: 0,
                  left: 0,
                  right: 0,
                  height: '6px',
                  backgroundColor: 'rgba(0,0,0,0.3)'
                }}>
                  <Box sx={{
                    height: '100%',
                    width: `${bufferStatus}%`,
                    backgroundColor: 'primary.main',
                    transition: 'width 0.3s ease'
                  }} />
                </Box>
              )}
            </Box>
          ) : (
            <Box sx={{ width: '100%', textAlign: 'center' }}>
              <input
                accept="video/*"
                style={{ display: 'none' }}
                id="video-upload"
                type="file"
                onChange={handleVideoFileChange}
              />
              <label htmlFor="video-upload">
                <Button 
                  variant="contained" 
                  component="span"
                  sx={{ mb: 2 }}
                >
                  Choose Video File
                </Button>
              </label>
              {videoFile && (
                <Box sx={{
                  width: '100%',
                  maxWidth: '480px',
                  mx: 'auto',
                  mt: 2
                }}>
                  <video
                    ref={videoRef}
                    src={URL.createObjectURL(videoFile)}
                    controls
                    style={{
                      width: '100%',
                      borderRadius: '4px',
                      border: '2px solid #ddd'
                    }}
                  />
                </Box>
              )}
            </Box>
          )}
        </Box>

        {/* Action buttons */}
        <Box sx={{ textAlign: 'center', mb: 3 }}>
          {inputType === 'webcam' ? (
            <>
              <Button
                variant="contained"
                size="large"
                onClick={startWebcamPredictions}
                disabled={isProcessing || !isWebcamActive || connectionStatus !== 'connected'}
                startIcon={isProcessing ? <CircularProgress size={20} color="inherit" /> : null}
                sx={{
                  px: 4,
                  py: 1.5,
                  fontSize: '1.1rem',
                  mr: 2
                }}
              >
                {isProcessing ? 'Processing' : 'Start Recognition'}
              </Button>
              <Button
                variant="outlined"
                size="large"
                onClick={stopWebcamPredictions}
                disabled={!isWebcamActive || !frameInterval.current}
                sx={{
                  px: 4,
                  py: 1.5,
                  fontSize: '1.1rem'
                }}
              >
                Stop Recognition
              </Button>
              {connectionStatus.includes('reconnecting') && (
                <Button
                  variant="text"
                  size="large"
                  onClick={() => {
                    reconnectAttempt.current = 0;
                    connectWebSocket();
                  }}
                  startIcon={<ReplayIcon />}
                  sx={{
                    ml: 2,
                    color: 'warning.main'
                  }}
                >
                  Reconnect Now
                </Button>
              )}
            </>
          ) : (
            <Button
              variant="contained"
              size="large"
              onClick={processVideo}
              disabled={isProcessing || !videoFile}
              startIcon={isProcessing ? <CircularProgress size={20} color="inherit" /> : null}
              sx={{
                px: 4,
                py: 1.5,
                fontSize: '1.1rem'
              }}
            >
              {isProcessing ? 'Processing' : 'Process Video'}
            </Button>
          )}
        </Box>

        {bufferStatus > 0 && bufferStatus < 100 && (
          <Typography variant="body2" sx={{ mb: 2, color: 'text.secondary' }}>
            Collecting frames: {bufferStatus}% complete
          </Typography>
        )}

        {error && (
          <Paper elevation={2} sx={{ p: 3, backgroundColor: '#f5f5f5', mb: 3 }}>
            <Typography variant="h6" sx={{ mb: 1 }}>
              Error:
            </Typography>
            <Typography variant="body1" sx={{ 
              fontSize: '1.2rem',
              fontWeight: 'medium'
            }}>
              {error}
            </Typography>
          </Paper>
        )}

        {prediction && (
          <Paper elevation={2} sx={{ p: 3, backgroundColor: '#f5f5f5' }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <Typography variant="h6" sx={{ mb: 1 }}>
                Prediction:
              </Typography>
              <IconButton 
                onClick={() => speakPrediction(prediction)} 
                disabled={isMuted}
                color="primary"
              >
                <VolumeUpIcon />
              </IconButton>
            </Box>
            <Typography variant="body1" sx={{ 
              fontSize: '1.2rem',
              fontWeight: 'medium'
            }}>
              {prediction}
            </Typography>
            <Typography variant="body2" sx={{ 
              fontSize: '1rem',
              fontWeight: 'medium'
            }}>
              Confidence: {(confidence * 100).toFixed(2)}%
            </Typography>
          </Paper>
        )}
      </Paper>
    </Container>
  );
};

export default SignToSpeech;