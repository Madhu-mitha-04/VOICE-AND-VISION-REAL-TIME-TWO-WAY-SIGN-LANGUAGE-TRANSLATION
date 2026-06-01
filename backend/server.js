const express = require('express');
const mysql = require('mysql2/promise');
const cors = require('cors');
const fs = require('fs');
const path = require('path');

const app = express();
app.use(cors());
app.use(express.json());

// Database connection pool
const pool = mysql.createPool({
  host: 'localhost',
  user: 'root',
  password: 'Madhu@pro',
  database: 'speech_to_sign',
  waitForConnections: true,
  connectionLimit: 10,
  queueLimit: 0
});

// API endpoints
app.get('/videos', async (req, res) => {
  try {
    const [videos] = await pool.query('SELECT * FROM videos');
    res.json(videos);
  } catch (err) {
    console.error('Database error:', err);
    res.status(500).json({ error: 'Database error' });
  }
});

// Dynamic action matching endpoint
app.get('/match-action', async (req, res) => {
  const searchText = (req.query.text || '').toLowerCase().replace(/\s+/g, '');
  
  try {
    const [videos] = await pool.query('SELECT * FROM videos');
    
    // Find best match (checks exact, contains, and similarity)
    const matchedVideo = videos.find(video => {
      const normalizedAction = video.word.toLowerCase().replace(/\s+/g, '');
      return (
        searchText === normalizedAction || 
        searchText.includes(normalizedAction) ||
        normalizedAction.includes(searchText) ||
        calculateSimilarity(searchText, normalizedAction) > 0.7
      );
    });

    if (matchedVideo) {
      // Verify video file exists
      if (!fs.existsSync(matchedVideo.video_path)) {
        return res.status(404).json({ 
          success: false,
          message: 'Video file not found on server' 
        });
      }

      return res.json({
        success: true,
        action: matchedVideo.word,
        videoUrl: `/stream-video/${matchedVideo.id}`
      });
    }
    
    res.status(404).json({ success: false, message: 'No matching sign found' });
  } catch (err) {
    console.error('Matching error:', err);
    res.status(500).json({ error: 'Server error' });
  }
});

// Helper function for similarity matching
function calculateSimilarity(str1, str2) {
  const longer = str1.length > str2.length ? str1 : str2;
  const shorter = str1.length > str2.length ? str2 : str1;
  const longerLength = longer.length;
  if (longerLength === 0) return 1.0;
  return (longerLength - editDistance(longer, shorter)) / parseFloat(longerLength);
}

// Helper function for edit distance
function editDistance(s1, s2) {
  s1 = s1.toLowerCase();
  s2 = s2.toLowerCase();
  const costs = [];
  for (let i = 0; i <= s1.length; i++) {
    let lastValue = i;
    for (let j = 0; j <= s2.length; j++) {
      if (i === 0) costs[j] = j;
      else {
        if (j > 0) {
          let newValue = costs[j - 1];
          if (s1.charAt(i - 1) !== s2.charAt(j - 1))
            newValue = Math.min(Math.min(newValue, lastValue), costs[j]) + 1;
          costs[j - 1] = lastValue;
          lastValue = newValue;
        }
      }
    }
    if (i > 0) costs[s2.length] = lastValue;
  }
  return costs[s2.length];
}

// Video streaming endpoint
app.get('/stream-video/:id', async (req, res) => {
  try {
    const videoId = req.params.id;
    const [results] = await pool.query(
      'SELECT video_path FROM videos WHERE id = ?', 
      [videoId]
    );

    if (!results.length) return res.status(404).send('Video not found');

    const videoPath = path.join(__dirname, results[0].video_path);
    if (!fs.existsSync(videoPath)) {
      return res.status(404).send('Video file missing');
    }

    res.setHeader('Content-Type', 'video/mp4');
    const videoStream = fs.createReadStream(videoPath);
    videoStream.pipe(res);
    
    videoStream.on('error', (err) => {
      console.error('Stream error:', err);
      res.status(500).end();
    });

  } catch (err) {
    console.error('Streaming error:', err);
    res.status(500).send('Server error');
  }
});

const PORT = 3001;
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
  console.log(`Match endpoint: http://localhost:${PORT}/match-action?text=thankyou`);
});