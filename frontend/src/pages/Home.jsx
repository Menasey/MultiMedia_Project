import React from 'react';
import { Link } from 'react-router-dom';

function Home() {
  return (
    <>
      <div className="navbar">
        <Link to="/login">Login</Link>
        <Link to="/register" style={{ marginLeft: '1rem' }}>Register</Link>
      </div>

      <div className="container">
        <h1 className="title">Welcome to Crisis Classifier</h1>
        <p className="subtitle">
          This web app allows administrators to train models on crisis event URLs,
          and users to classify new webpages as <strong>Crisis</strong> or <strong>Non-Crisis</strong>.
        </p>
        <ul className="list">
          <li><strong>Admins:</strong> Upload URL batches and train models.</li>
          <li><strong>Users:</strong> Classify URLs with trained models.</li>
        </ul>
      </div>
    </>
  );
}

export default Home;
