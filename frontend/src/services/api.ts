import axios from 'axios';

const api = axios.create({
  baseURL: (import.meta as any).env.VITE_API_URL || 'http://127.0.0.1:8000/api',
});

export const assessRisk = async (data: any) => {
  const response = await api.post('/assessment/predict', data);
  return response.data;
};

export const getPatientTemporalProfile = async (patientId: string) => {
  const response = await api.get(`/patients/${patientId}/temporal`);
  return response.data;
};

export const getAnalytics = async () => {
  const response = await api.get('/analytics');
  return response.data;
};

export const getIncrementalOverview = async () => {
  const response = await api.get('/incremental/overview');
  return response.data;
};

export const validateIncrementalBatch = async (clinicalFile: File, temporalFile?: File) => {
  const data = new FormData();
  data.append('clinical_file', clinicalFile);
  if (temporalFile) data.append('temporal_file', temporalFile);
  const response = await api.post('/incremental/batches/validate', data);
  return response.data;
};

export const confirmIncrementalBatch = async (batchId: string) => {
  const response = await api.post(`/incremental/batches/${encodeURIComponent(batchId)}/confirm`);
  return response.data;
};

export const trainIncrementalChallenger = async (idempotencyKey?: string) => {
  const key = idempotencyKey || (typeof crypto !== 'undefined' && crypto.randomUUID ? crypto.randomUUID() : `ilkey-${Date.now()}-${Math.random().toString(36).slice(2, 11)}`);
  const response = await api.post('/incremental/training-runs', null, {
    headers: { 'Idempotency-Key': key },
  });
  return response.data;
};

export const promoteIncrementalChallenger = async (runId: string) => {
  const response = await api.post(`/incremental/training-runs/${encodeURIComponent(runId)}/promote`);
  return response.data;
};

export const rejectIncrementalChallenger = async (runId: string, reason: string) => {
  const response = await api.post(`/incremental/training-runs/${encodeURIComponent(runId)}/reject`, { reason });
  return response.data;
};

export const rollbackIncrementalChampion = async () => {
  const response = await api.post('/incremental/champion/rollback');
  return response.data;
};

export const getRunSyntheticCount = async (runId: string) => {
  const response = await api.get(`/incremental/training-runs/${encodeURIComponent(runId)}/synthetic-count`);
  return response.data;
};

export default api;
