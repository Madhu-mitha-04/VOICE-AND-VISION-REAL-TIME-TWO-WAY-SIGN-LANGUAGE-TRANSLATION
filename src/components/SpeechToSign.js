import React, { useState, useEffect, useRef } from 'react';

const SpeechToSign = () => {
    const [videoUrl, setVideoUrl] = useState('');
    const [action, setAction] = useState('');
    const [isListening, setIsListening] = useState(false);
    const [status, setStatus] = useState('Ready');
    const [inputText, setInputText] = useState('');
    const [inputMode, setInputMode] = useState('text'); // 'text' or 'speech'
    const videoRef = useRef(null);
    const recognitionRef = useRef(null);

    // Initialize speech recognition
    useEffect(() => {
        if (inputMode !== 'speech') return;

        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        if (!SpeechRecognition) {
            setStatus('Speech recognition not supported');
            return;
        }

        recognitionRef.current = new SpeechRecognition();
        recognitionRef.current.continuous = false;
        recognitionRef.current.interimResults = false;
        recognitionRef.current.maxAlternatives = 3;

        recognitionRef.current.onstart = () => {
            setStatus('Listening... Speak now');
        };

        recognitionRef.current.onresult = async (event) => {
            const transcript = event.results[0][0].transcript.toLowerCase();
            setStatus(`Recognized: ${transcript}`);
            setInputText(transcript);
            await processInput(transcript);
            setIsListening(false);
        };

        recognitionRef.current.onerror = (event) => {
            console.error('Recognition error:', event.error);
            setStatus(`Error: ${event.error}`);
            setIsListening(false);
        };

        return () => {
            recognitionRef.current?.stop();
        };
    }, [inputMode]);

    const processInput = async (text) => {
        try {
            const response = await fetch(
                `http://localhost:3001/match-action?text=${encodeURIComponent(text)}`
            );
            const data = await response.json();
            
            if (data.success) {
                setAction(data.action);
                const videoUrl = `http://localhost:3001${data.videoUrl}?t=${Date.now()}`;
                setVideoUrl(videoUrl);
                
                if (videoRef.current) {
                    videoRef.current.load();
                    videoRef.current.play().catch(e => {
                        console.error('Play error:', e);
                    
                    });
                }
            } else {
                setStatus(data.message || 'No matching sign found');
            }
        } catch (error) {
            console.error('API error:', error);
            setStatus('Network error');
        }
    };

    const toggleListening = () => {
        if (inputMode !== 'speech') return;
        
        if (!isListening) {
            recognitionRef.current.start();
            setIsListening(true);
            setAction('');
            setVideoUrl('');
            setStatus('Listening... Speak now');
        }
    };

    const handleTextSubmit = async (e) => {
        e.preventDefault();
        if (!inputText.trim()) return;
        
        setStatus('Processing text...');
        await processInput(inputText);
    };

    return (
        <div style={{ 
            textAlign: 'center', 
            padding: '20px',
            maxWidth: '800px',
            margin: '0 auto'
        }}>
            <h2>Speech to Sign</h2>
            
            <div style={{ margin: '20px 0' }}>
                <div style={{ display: 'flex', justifyContent: 'center', gap: '20px', marginBottom: '20px' }}>
                    <button 
                        onClick={() => setInputMode('text')}
                        style={{
                            padding: '10px 20px',
                            backgroundColor: inputMode === 'text' ? '#4CAF50' : '#e0e0e0',
                            color: inputMode === 'text' ? 'white' : 'black',
                            border: 'none',
                            borderRadius: '5px',
                            cursor: 'pointer'
                        }}
                    >
                        Text Input
                    </button>
                    <button 
                        onClick={() => setInputMode('speech')}
                        style={{
                            padding: '10px 20px',
                            backgroundColor: inputMode === 'speech' ? '#4CAF50' : '#e0e0e0',
                            color: inputMode === 'speech' ? 'white' : 'black',
                            border: 'none',
                            borderRadius: '5px',
                            cursor: 'pointer'
                        }}
                    >
                        Speech Input
                    </button>
                </div>

                {inputMode === 'text' ? (
                    <form onSubmit={handleTextSubmit} style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                        <textarea
                            value={inputText}
                            onChange={(e) => setInputText(e.target.value)}
                            placeholder="Enter text to convert to sign language"
                            style={{
                                width: '100%',
                                minHeight: '100px',
                                padding: '10px',
                                fontSize: '16px',
                                marginBottom: '10px',
                                borderRadius: '5px',
                                border: '1px solid #ccc'
                            }}
                        />
                        <button 
                            type="submit"
                            style={{
                                padding: '12px 24px',
                                fontSize: '18px',
                                backgroundColor: '#4CAF50',
                                color: 'white',
                                border: 'none',
                                borderRadius: '5px',
                                cursor: 'pointer',
                                width: '200px'
                            }}
                        >
                            Convert
                        </button>
                    </form>
                ) : (
                    <button 
                        onClick={toggleListening}
                        disabled={isListening}
                        style={{
                            padding: '12px 24px',
                            fontSize: '18px',
                            backgroundColor: isListening ? '#cccccc' : '#4CAF50',
                            color: 'white',
                            border: 'none',
                            borderRadius: '5px',
                            cursor: 'pointer',
                            width: '200px'
                        }}
                    >
                        {isListening ? 'Listening...' : 'Start Listening'}
                    </button>
                )}
            </div>

            <div style={{ margin: '20px 0' }}>
                <p style={{ fontSize: '18px' }}>
                    <strong>Status:</strong> {status}
                </p>
                {action && (
                    <p style={{ fontSize: '24px', fontWeight: 'bold', marginTop: '10px' }}>
                        Action: {action}
                    </p>
                )}
            </div>

            <div style={{ 
                width: '100%',
                borderRadius: '8px',
                overflow: 'hidden',
                boxShadow: '0 4px 8px rgba(0,0,0,0.1)'
            }}>
                <video
                    ref={videoRef}
                    src={videoUrl}
                    width="100%"
                    autoPlay
                    muted
                    playsInline
                    onError={(e) => {
                        console.error('Video error:', e);
                        setStatus('animation loading')
                    }}
                    style={{ 
                        display: videoUrl ? 'block' : 'none',
                        backgroundColor: '#000'
                    }}
                />
            </div>
        </div>
    );
};

export default SpeechToSign;