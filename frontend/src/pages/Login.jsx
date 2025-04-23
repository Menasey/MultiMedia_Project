import React, { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { loginUser } from '../services/api';

function Login() {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [msg, setMsg] = useState('');
  const navigate = useNavigate();

  async function handleSubmit(e) {
    e.preventDefault();
    try {
      const res = await loginUser(username, password);
      localStorage.setItem('token', res.access_token);
      navigate(res.is_admin ? '/admin' : '/user');
    } catch (err) {
      setMsg('Login failed: ' + (err.response?.data?.detail || 'Unknown error'));
    }
  }

  return (
    <>
      <div className="navbar">
        <Link to="/">Home</Link>
        <Link to="/register">Register</Link>
      </div>

      <form onSubmit={handleSubmit} className="form">
        <h1 className="title">Login</h1>
        <input type="text" className="input" placeholder="Username" value={username} onChange={e => setUsername(e.target.value)} required />
        <input type="password" className="input" placeholder="Password" value={password} onChange={e => setPassword(e.target.value)} required />
        <button type="submit" className="button">Log In</button>
        {msg && <div style={{color:'red', marginTop:'1rem', textAlign:'center'}}>{msg}</div>}
      </form>
    </>
  );
}

export default Login;
