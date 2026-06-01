import { Container, Typography, Box, Button, Grid } from '@mui/material';
import { Link } from 'react-router-dom';
import './Home.css'; // Import custom CSS for animations
import { FaSignLanguage, FaMicrophone } from 'react-icons/fa'; // Import icons

const Home = () => {
  return (
    <Container
      sx={{
        textAlign: 'center',
        padding: '50px',
        minHeight: '100vh',
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'center',
        alignItems: 'center',
        background: 'white', // Set background to white
        color: 'black', // Set text color to black
      }}
    >
      <Box
        sx={{
          animation: 'fadeIn 2s',
        }}
      >
        <Typography
          variant="h2"
          gutterBottom
          sx={{
            fontFamily: 'Pacifico, cursive',
            animation: 'slideIn 1s',
            fontWeight: 'bold',
            fontSize: { xs: '2.5rem', sm: '3.5rem', md: '4rem' },
            color: '#000000', // Black text color
          }}
        >
          Welcome to Voice & Vision
        </Typography>
        <Typography
          variant="h5"
          sx={{
            fontFamily: 'Open Sans, sans-serif',
            animation: 'fadeIn 2s',
            marginBottom: '40px',
            fontSize: { xs: '1.2rem', sm: '1.5rem', md: '1.8rem' },
            color: '#000000', // Black text color
          }}
        >
          A real-time two-way sign language translation system for deaf, mute, and blind individuals.
        </Typography>
        <Grid container spacing={4} justifyContent="center">
          <Grid item xs={12} sm={6} md={4}>
            <Box
              sx={{
                backgroundColor: '#25acee', // Cream color for boxes
                padding: '30px',
                borderRadius: '15px',
                boxShadow: '0 8px 16px rgba(14, 0, 0, 0.1)',
                transition: 'transform 0.3s ease, box-shadow 0.3s ease',
                '&:hover': {
                  transform: 'scale(1.05)',
                  boxShadow: '0 12px 24px rgba(14, 0, 0, 0.1)',
                },
              }}
            >
              <FaSignLanguage size={60} color="" /> {/* Pastel icon color */}
              <Typography variant="h6" sx={{ margin: '20px 0', fontWeight: 'bold', color: '#000000' }}>
                Sign to Speech
              </Typography>
              <Button
                variant="contained"
                color="primary"
                component={Link}
                to="/sign-to-speech"
                sx={{
                  backgroundColor: '', // Pastel button color
                  color: 'white',
                  '&:hover': { backgroundColor: '' }, // Darker pastel on hover
                  transition: 'background-color 0.3s ease',
                  padding: '10px 30px',
                  fontSize: '16px',
                }}
              >
                Get Started
              </Button>
            </Box>
          </Grid>
          <Grid item xs={12} sm={6} md={4}>
            <Box
              sx={{
                backgroundColor: '#25acee', // Cream color for boxes
                padding: '30px',
                borderRadius: '15px',
                boxShadow: '0 8px 16px rgba(14, 0, 0, 0.1)',
                transition: 'transform 0.3s ease, box-shadow 0.3s ease',
                '&:hover': {
                  transform: 'scale(1.05)',
                  boxShadow: '0 12px 24px rgba(14, 0, 0, 0.1)',
                },
              }}
            >
              <FaMicrophone size={60} color="" /> {/* Pastel icon color */}
              <Typography variant="h6" sx={{ margin: '20px 0', fontWeight: 'bold', color: '#000000' }}>
                Speech to Sign
              </Typography>
              <Button
                variant="contained"
                color="secondary"
                component={Link}
                to="/speech-to-sign"
                sx={{
                  backgroundColor: 'rgb(14, 127, 219) ', // Pastel button color
                  color: 'white',
                  '&:hover': { backgroundColor: '#1565C0 ' }, // Darker pastel on hover
                  transition: 'background-color 0.3s ease',
                  padding: '10px 30px',
                  fontSize: '16px',
                }}
              >
                Get Started
              </Button>
            </Box>
          </Grid>
        </Grid>
      </Box>
    </Container>
  );
};

export default Home;
