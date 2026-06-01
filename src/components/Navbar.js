import { AppBar, Toolbar, Typography, Button } from '@mui/material';
import { Link } from 'react-router-dom';

const Navbar = () => {
  return (
    <AppBar position="static" sx={{ backgroundColor: '', boxShadow: '0 4px 8px rgba(0, 0, 0, 0.2)' }}>
      <Toolbar>
        <Typography variant="h6" sx={{ flexGrow: 1, fontFamily: 'Pacifico, cursive' }}>
          Voice & Vision
        </Typography>
        <Button
          color="inherit"
          component={Link}
          to="/"
          sx={{ '&:hover': { color: '#ff4081' }, transition: 'color 0.3s ease' }}
        >
          Home
        </Button>
        <Button
          color="inherit"
          component={Link}
          to="/sign-to-speech"
          sx={{ '&:hover': { color: '#ff4081' }, transition: 'color 0.3s ease' }}
        >
          Sign to Speech
        </Button>
        <Button
          color="inherit"
          component={Link}
          to="/speech-to-sign"
          sx={{ '&:hover': { color: '#ff4081' }, transition: 'color 0.3s ease' }}
        >
          Speech to Sign
        </Button>
      </Toolbar>
    </AppBar>
  );
};

export default Navbar;