import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import Navbar from './components/Navbar';
import Home from './components/Home';
import SignToSpeech from './components/SignToSpeech';
import SpeechToSign from './components/SpeechToSign';

function App() {
  return (
    <Router>
      <Navbar />
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/sign-to-speech" element={<SignToSpeech />} />
        <Route path="/speech-to-sign" element={<SpeechToSign />} />
      </Routes>
    </Router>
  );
}

export default App;