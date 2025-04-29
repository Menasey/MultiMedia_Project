import React from 'react';
import { useLocation, useNavigate } from 'react-router-dom';

function ModelEvaluation() {
  const location = useLocation();
  const navigate = useNavigate();
  const model = location.state?.model;

  if (!model) {
    return (
      <div className="container">
        <h1 className="title">No Model Selected</h1>
        <button onClick={() => navigate('/admin')} className="button">
          Back to Admin Dashboard
        </button>
      </div>
    );
  }

  return (
    <div className="container">
      <h1 className="title">{model.name} ({model.classifier})</h1>
      <p className="subtitle">Trained on {new Date(model.training_date).toLocaleString()}</p>

      {model.eval_score_plot && (
        <>
          <h2 className="subtitle">Decision Score Distribution</h2>
          <img src={model.eval_score_plot} alt="Decision Scores" style={{ width: '100%', maxWidth: '700px', marginBottom: '2rem' }} />
        </>
      )}

      {model.eval_terms_plot && (
        <>
          <h2 className="subtitle">Top TF-IDF Terms</h2>
          <img src={model.eval_terms_plot} alt="TFIDF Terms" style={{ width: '100%', maxWidth: '700px' }} />
        </>
      )}

      {model.eval_fold_plot && (
        <>
          <h2 className="subtitle">Cross-Validation Scores per Fold</h2>
          <img src={model.eval_fold_plot} alt="Cross-Validation" style={{ width: '100%', maxWidth: '700px' }} />
        </>
        )}

      <button onClick={() => navigate('/admin')} className="button" style={{ marginTop: '2rem' }}>
        Back to Admin Dashboard
      </button>
    </div>
  );
}

export default ModelEvaluation;
