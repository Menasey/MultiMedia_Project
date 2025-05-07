import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  getCollections,
  getModels,
  createCollection,
  uploadUrlsToCollection,
  getCollectionDetails,
  deleteCollection,
} from '../services/api';

function UserDashboard() {
  const [collections, setCollections] = useState([]);
  const [models, setModels] = useState([]);
  const [newTitle, setNewTitle] = useState('');
  const [newModelId, setNewModelId] = useState('');
  const [newZip, setNewZip] = useState(null);
  const [expanded, setExpanded] = useState({});
  const [collectionDetails, setCollectionDetails] = useState({});
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState('');
  const navigate = useNavigate();

  useEffect(() => {
    loadCollections();
    loadModels();
  }, []);

  async function loadCollections() {
    const data = await getCollections();
    setCollections(data);
    const init = {};
    data.forEach(c => (init[c.id] = false));
    setExpanded(init);
  }

  async function loadModels() {
    setModels(await getModels());
  }

  async function handleNewCollection(e) {
    e.preventDefault();
    if (!newZip) return;
    setLoading(true);
    setStatus('Creating collection and uploading...');

    try {
      const coll = await createCollection(newTitle, newModelId);
      await uploadUrlsToCollection(coll.id, newZip);
      await loadCollections();
      setStatus('✅ Collection created and classified!');
      setNewTitle('');
      setNewModelId('');
      setNewZip(null);
    } catch (err) {
      console.error('Error creating collection:', err);
      setStatus('❌ Failed to create collection.');
    } finally {
      setLoading(false);
    }
  }

  async function toggleCollection(id) {
    if (!expanded[id]) {
      const data = await getCollectionDetails(id);
      setCollectionDetails(prev => ({ ...prev, [id]: data }));
    }
    setExpanded(prev => ({ ...prev, [id]: !prev[id] }));
  }

  async function handleDeleteCollection(id) {
    if (window.confirm("Delete this collection?")) {
      await deleteCollection(id);
      await loadCollections();
    }
  }

  function handleViewModelVisualizations(collection) {
    const model = models.find(m => m.id === collection.model_id);
    if (model) {
      navigate('/model-eval', { state: { model, fromAdmin: false } });
    } else {
      alert('Model not found.');
    }
  }

  function renderPrediction(pred, score) {
    const isCrisis = pred === 'Crisis';

    return (
      <span style={{ color: isCrisis ? 'green' : 'red', fontWeight: 'bold' }}>
        {isCrisis ? 'Crisis Related' : 'Non-Crisis Related'} 
      </span>
    );
  }

  return (
    <>
      <div className="navbar">
        <button onClick={() => { localStorage.removeItem('token'); navigate('/login'); }} className="logout-button">
          Logout
        </button>
      </div>

      <div className="container">
        <h1 className="title">Your Collections</h1>

        <form onSubmit={handleNewCollection} className="form">
          <h2 className="subtitle">Create New Collection</h2>

          <input
            type="text"
            className="input"
            placeholder="Collection Title"
            value={newTitle}
            onChange={e => setNewTitle(e.target.value)}
            required
          />

          <select
            className="input"
            value={newModelId}
            onChange={e => setNewModelId(e.target.value)}
            required
          >
            <option value="">Select Model</option>
            {models.map(m => (
              <option key={m.id} value={m.id}>{m.name} ({m.classifier})</option>
            ))}
          </select>

          <input
            type="file"
            accept=".zip"
            className="input"
            onChange={e => setNewZip(e.target.files[0])}
            required
          />

          <button type="submit" className="button">Create and Classify</button>
        </form>

        {loading && <div className="spinner" />}
        {status && <div style={{ marginTop: '1rem', color: '#2563eb' }}>{status}</div>}

        <h2 className="subtitle" style={{ marginTop: '2rem' }}>Existing Collections</h2>

        <ul className="list">
          {collections.map(c => (
            <li key={c.id} className="collection-item">
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <strong>{c.title}</strong>
                <button
                  onClick={() => handleDeleteCollection(c.id)}
                  style={{
                    background: 'transparent',
                    border: 'none',
                    fontSize: '1.2rem',
                    color: '#e11d48',
                    cursor: 'pointer',
                  }}
                  title="Delete Collection"
                >
                  🗑️
                </button>
              </div>

              <button className="button" onClick={() => toggleCollection(c.id)} style={{ marginTop: '0.5rem' }}>
                {expanded[c.id] ? 'Hide Details' : 'Show Details'}
              </button>

              {expanded[c.id] && collectionDetails[c.id] && (
                <div style={{ marginTop: '1rem' }}>
                  <button className="button" onClick={() => handleViewModelVisualizations(collectionDetails[c.id])}>
                    View Model Visualizations
                  </button>

                  <ul className="list" style={{ marginTop: '1rem' }}>
                    {collectionDetails[c.id].items.map((item, idx) => (
                      <li key={idx} className="collection-item">
                        <a
                          href={item.url}
                          target="_blank"
                          rel="noopener noreferrer"
                          style={{ fontWeight: 'bold', wordBreak: 'break-all' }}
                        >
                          {item.url}
                        </a>
                        <div>{renderPrediction(item.pred, item.score)}</div>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </li>
          ))}
        </ul>
      </div>
    </>
  );
}

export default UserDashboard;