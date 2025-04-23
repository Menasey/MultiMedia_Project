import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { getCollections, getModels, createCollection, uploadUrlsToCollection, getCollectionDetails } from '../services/api';

function UserDashboard() {
  const [collections, setCollections] = useState([]);
  const [models, setModels] = useState([]);
  const [newTitle, setNewTitle] = useState('');
  const [newModelId, setNewModelId] = useState('');
  const [newZip, setNewZip] = useState(null);
  const [activeCollection, setActiveCollection] = useState(null);
  const navigate = useNavigate();

  function logout() {
    localStorage.removeItem('token');
    navigate('/login');
  }

  async function loadCollections() {
    const data = await getCollections();
    setCollections(data);
  }

  async function loadModels() {
    const data = await getModels();
    setModels(data);
  }

  useEffect(() => {
    loadCollections();
    loadModels();
  }, []);

  async function handleNewCollection(e) {
    e.preventDefault();
    if (!newZip) return;
    const coll = await createCollection(newTitle, newModelId);
    await uploadUrlsToCollection(coll.id, newZip);
    await loadCollections();
  }

  async function openCollection(id) {
    const data = await getCollectionDetails(id);
    setActiveCollection(data);
  }

  return (
    <>
      <div className="navbar">
        <button onClick={logout} className="logout-button">Logout</button>
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
              <option key={m.id} value={m.id}>{m.name}</option>
            ))}
          </select>
          <input
            type="file"
            accept=".zip"
            className="input"
            onChange={e => setNewZip(e.target.files[0])}
            required
          />
          <button type="submit" className="button">
            Create and Classify
          </button>
        </form>

        <h2 className="subtitle" style={{ marginTop: '2rem' }}>Existing Collections</h2>
        <ul className="list">
          {collections.map(c => (
            <li key={c.id} className="collection-item" onClick={() => openCollection(c.id)}>
              {c.title}
            </li>
          ))}
        </ul>

        {activeCollection && (
          <div style={{ marginTop: '2rem' }}>
            <h3 className="subtitle">{activeCollection.title}</h3>
            <ul className="list">
              {activeCollection.items.map((i, idx) => (
                <li key={idx} className="collection-item">
                  <div style={{ fontFamily: 'monospace', wordBreak: 'break-all' }}>{i.url}</div>
                  <div>{i.pred} ({i.score.toFixed(2)})</div>
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </>
  );
}

export default UserDashboard;
