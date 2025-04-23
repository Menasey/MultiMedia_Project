import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { registerUser, loginUser } from '../services/api';

function Register() {
  const [username, setUsername] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [isAdmin, setIsAdmin] = useState(false);
  const [msg, setMsg] = useState('');
  const navigate = useNavigate();

  async function handleSubmit(e) {
    e.preventDefault();
    try {
      await registerUser({ username, email, password, is_admin: isAdmin });
      const res = await loginUser(username, password);
      localStorage.setItem('token', res.access_token);
      navigate(res.is_admin ? '/admin' : '/user');
    } catch (err) {
      setMsg('Registration failed: ' + (err.response?.data?.detail || 'Unknown error'));
    }
  }

  return (
    <>
      <div className="navbar">
        <Link to="/">Home</Link>
        <Link to="/login">Login</Link>
      </div>

      <form onSubmit={handleSubmit} className="form">
        <h1 className="title">Register</h1>
        <input type="text" className="input" placeholder="Username" value={username} onChange={e => setUsername(e.target.value)} required />
        <input type="email" className="input" placeholder="Email" value={email} onChange={e => setEmail(e.target.value)} required />
        <input type="password" className="input" placeholder="Password" value={password} onChange={e => setPassword(e.target.value)} required />
        <label>
          <input type="checkbox" checked={isAdmin} onChange={e => setIsAdmin(e.target.checked)} /> Register as Admin
        </label>
        <button type="submit" className="button">Sign Up</button>
        {msg && <div style={{color:'red', marginTop:'1rem', textAlign:'center'}}>{msg}</div>}
      </form>
    </>
  );
}

export default Register;
