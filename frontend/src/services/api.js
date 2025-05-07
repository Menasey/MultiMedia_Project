import axios from 'axios';

const API_URL = 'http://127.0.0.1:8000';  // Your FastAPI backend

// Axios instance for authenticated requests
const authAxios = axios.create({
  baseURL: API_URL,
});

// Automatically attach token to all requests
authAxios.interceptors.request.use(config => {
  const token = localStorage.getItem('token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// === Auth related ===
export async function loginUser(username, password) {
  const params = new URLSearchParams();
  params.append('username', username);
  params.append('password', password);

  const response = await axios.post(`${API_URL}/token`, params, {
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' }
  });
  return response.data;  // { access_token, token_type, is_admin }
}

export async function registerUser({ username, email, password, is_admin }) {
  const formData = new FormData();
  formData.append('username', username);
  formData.append('email', email);
  formData.append('password', password);
  formData.append('is_admin', is_admin);

  const response = await axios.post(`${API_URL}/register`, formData);
  return response.data;  // { msg: "Registered successfully" }
}

// === Models (Admin side) ===
export async function getModels() {
  const response = await authAxios.get('/models');
  return response.data.map(model => ({
    ...model,
    eval_score_plot: model.eval_score_plot ? `${API_URL}/${model.eval_score_plot}` : null,
    eval_terms_plot: model.eval_terms_plot ? `${API_URL}/${model.eval_terms_plot}` : null,
    eval_fold_plot: model.eval_fold_plot ? `${API_URL}/${model.eval_fold_plot}` : null,
  }));
}

export async function trainModel(zipFile, modelName, classifier = 'svm', description) {
  const formData = new FormData();
  formData.append('zip_file', zipFile);
  formData.append('model_name', modelName);
  formData.append('classifier', classifier);
  formData.append('description', description);

  const response = await authAxios.post('/models/train', formData);
  return response.data; // { job_id }
}

export async function getTrainingJobStatus(jobId) {
  const res = await authAxios.get(`/training_jobs/${jobId}`);
  return res.data;
}

export async function deleteModel(modelId) {
  const response = await authAxios.delete(`/models/${modelId}`);
  return response.data; // { detail: "Model deleted successfully" }
}

export async function deleteAllModels() {
  const response = await authAxios.delete('/models');
  return response.data; // { detail: "All models and related data deleted" }
}

// === Collections (User side) ===
export async function createCollection(title, modelId) {
  const params = new URLSearchParams();
  params.append('title', title);
  params.append('model_id', modelId);

  const response = await authAxios.post('/collections', params);
  return response.data; // collection info
}

export async function getCollections() {
  const response = await authAxios.get('/collections');
  return response.data; // list of collections
}

export async function uploadUrlsToCollection(collectionId, zipFile) {
  const formData = new FormData();
  formData.append('file', zipFile);

  const response = await authAxios.post(`/collections/${collectionId}/predict`, formData);
  return response.data; // { added: number }
}

export async function getCollectionDetails(collectionId) {
  const response = await authAxios.get(`/collections/${collectionId}`);
  return response.data; // { id, title, items: [...] }
}

export async function deleteCollection(collectionId) {
  const res = await authAxios.delete(`/collections/${collectionId}`);
  return res.data;
}

