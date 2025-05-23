import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  getModels,
  trainModel,
  getTrainingJobStatus,
  deleteModel,
  deleteAllModels
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
  const [visiblePlots, setVisiblePlots] = useState({});

  const navigate = useNavigate();

  function logout() {
    localStorage.removeItem('token');
    navigate('/login');
  }

  async function loadModels() {
    const data = await getModels();
    setModels(data);
    const visibility = {};
    data.forEach(m => (visibility[m.id] = false));
    setVisiblePlots(visibility);
  }

  useEffect(() => {
    loadModels();
  }, []);

  useEffect(() => {
    if (!jobId) return;

    const interval = setInterval(async () => {
      try {
        const statusData = await getTrainingJobStatus(jobId);
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
    }, 3000);

    return () => clearInterval(interval);
  }, [jobId]);

  async function handleTrain(e) {
    e.preventDefault();
    if (!zipFile || !modelName || !description) {
      setStatus('❗ Please fill out all fields and upload a ZIP file.');
      return;
    }

    try {
      setLoading(true);
      setStatus('Training started...');
      const { job_id } = await trainModel(zipFile, modelName, classifier, description);
      setJobId(job_id);
      setStatus('✅ Training submitted. Waiting for completion...');
    } catch (err) {
      console.error('Training error:', err);
      setStatus('❌ Training failed: ' + (err.response?.data?.detail || err.message || 'Unknown error'));
      setLoading(false);
    }
  }

  async function handleDeleteModel(id) {
    if (window.confirm("Delete this model?")) {
      await deleteModel(id);
      await loadModels();
    }
  }

  async function handleClearAll() {
    if (window.confirm("Are you sure you want to delete all models?")) {
      await deleteAllModels();
      await loadModels();
    }
  }

  function togglePlots(id) {
    setVisiblePlots(prev => ({ ...prev, [id]: !prev[id] }));
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
            placeholder="Short model description"
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
            <option value="deep_one_class">Deep One Class</option>
          </select>

          <button type="submit" className="button">Train Model</button>
        </form>

        {loading && <div className="spinner" />}
        {status && <div style={{ marginTop: '1rem', color: '#2563eb' }}>{status}</div>}

        <h2 className="subtitle" style={{ marginTop: '2rem' }}>Trained Models</h2>

        <button className="button danger" onClick={handleClearAll} style={{ marginBottom: '1rem' }}>
          🗑️ Clear All Models
        </button>

        <ul className="list">
          {models.map(m => (
            <li key={m.id} className="collection-item">
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <div>
                  <strong>{m.name}</strong> ({m.classifier})<br />
                  <small>Trained at {new Date(m.training_date).toLocaleString()}</small>
                </div>
                <button
                  onClick={() => handleDeleteModel(m.id)}
                  style={{
                    background: 'transparent',
                    border: 'none',
                    fontSize: '1.2rem',
                    color: '#e11d48',
                    cursor: 'pointer'
                  }}
                  title="Delete Model"
                >
                  🗑️
                </button>
              </div>

              <button
                className="button"
                onClick={() => togglePlots(m.id)}
                style={{ marginTop: '0.75rem' }}
              >
                {visiblePlots[m.id] ? 'Hide Visualizations' : 'Show Visualizations'}
              </button>

              {visiblePlots[m.id] && (
                <div>
                  {m.eval_score_plot && (
                    <img
                      src={m.eval_score_plot}
                      alt="Decision Score Distribution"
                      style={{ width: '100%', maxWidth: '600px', margin: '1rem 0' }}
                    />
                  )}
                  {m.eval_terms_plot && (
                    <img
                      src={m.eval_terms_plot}
                      alt="Top TF-IDF Terms"
                      style={{ width: '100%', maxWidth: '600px', marginBottom: '1rem' }}
                    />
                  )}
                  {m.eval_fold_plot && (
                    <img
                      src={m.eval_fold_plot}
                      alt="Fold Scores"
                      style={{ width: '100%', maxWidth: '600px', marginBottom: '1rem' }}
                    />
                  )}
                </div>
              )}
            </li>
          ))}
        </ul>
      </div>
    </>
  );
}

export default AdminDashboard;

