import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  getModels,
  trainModel,
  getTrainingJobStatus
} from '../services/api';

function AdminDashboard() {
  const [models, setModels] = useState([]);
  const [modelName, setModelName] = useState('');
  const [description, setDescription] = useState('');
  const [zipFile, setZipFile] = useState(null);
  const [classifier, setClassifier] = useState('svm');
  const [status, setStatus] = useState('');
  const [loading, setLoading] = useState(false);
  const [jobId, setJobId] = useState(null);

  const navigate = useNavigate();

  function logout() {
    localStorage.removeItem('token');
    navigate('/login');
  }

  async function loadModels() {
    const data = await getModels();
    setModels(data);
  }

  useEffect(() => {
    loadModels();
  }, []);

  // Polling effect
  useEffect(() => {
    if (!jobId) return;

    const interval = setInterval(async () => {
      try {
        const statusData = await getTrainingJobStatus(jobId);
        console.log('[Polling] Job status:', statusData.status);

        if (statusData.status === 'completed') {
          setStatus('✅ Model training complete!');
          setLoading(false);
          setJobId(null);
          loadModels();
          clearInterval(interval);
        } else if (statusData.status === 'failed') {
          setStatus('❌ Training failed: ' + (statusData.error || 'Unknown error'));
          setLoading(false);
          setJobId(null);
          clearInterval(interval);
        }
      } catch (err) {
        console.error('Polling error:', err);
        clearInterval(interval);
      }
    }, 3000); // every 3s

    return () => clearInterval(interval);
  }, [jobId]);

  async function handleTrain(e) {
    e.preventDefault();
    if (!zipFile) {
      setStatus('❗ Please upload a ZIP file!');
      return;
    }
    if (!modelName) {
      setStatus('❗ Please enter a model name!');
      return;
    }
    if (!description) {
      setStatus('❗ Please provide a short description for the model.');
      return;
    }

    try {
      setLoading(true);
      setStatus('Training started...');
      const { job_id } = await trainModel(zipFile, modelName, classifier, description);
      setJobId(job_id);
      setStatus('✅ Training submitted. Waiting for completion...');
    } catch (err) {
        console.error('Training error:', err); // Add this
        setStatus('❌ Training failed: ' + (err.response?.data?.detail || err.message || 'Unknown error'));
        setLoading(false);
      }
  }

  return (
    <>
      <div className="navbar">
        <button onClick={logout} className="logout-button">Logout</button>
      </div>

      <div className="container">
        <h1 className="title">Admin Panel: Train New Model</h1>

        <form onSubmit={handleTrain} className="form">
          <input
            type="text"
            className="input"
            placeholder="Model Name"
            value={modelName}
            onChange={e => setModelName(e.target.value)}
            required
          />

          <textarea
            className="input"
            placeholder="Short model description (e.g., Earthquake-related crisis pages)"
            value={description}
            onChange={e => setDescription(e.target.value)}
            required
          />

          <input
            type="file"
            accept=".zip"
            className="input"
            onChange={e => setZipFile(e.target.files[0])}
            required
          />

          <select
            className="input"
            value={classifier}
            onChange={e => setClassifier(e.target.value)}
          >
            <option value="svm">SVM</option>
            <option value="iforest">Isolation Forest</option>
          </select>

          <button type="submit" className="button">
            Train Model
          </button>
        </form>

        {loading && <div className="spinner" />}
        {status && <div style={{ marginTop: '1rem', color: '#2563eb' }}>{status}</div>}

        <h2 className="subtitle" style={{ marginTop: '2rem' }}>Trained Models</h2>
        <ul className="list">
          {models.map(m => (
            <li key={m.id} className="collection-item">
              <strong>{m.name}</strong> ({m.classifier}) — Trained at {new Date(m.training_date).toLocaleString()}
            </li>
          ))}
        </ul>
      </div>
    </>
  );
}

export default AdminDashboard;
