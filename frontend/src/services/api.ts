import axios from 'axios';

const api = axios.create({
  baseURL: (import.meta as any).env.VITE_API_URL || 'http://127.0.0.1:8000/api',
});

export const assessRisk = async (data: any) => {
  const response = await api.post('/assessment/predict', data);
  return response.data;
};

export const getHistory = async (patientCode: string) => {
  const response = await api.get(`/history/${patientCode}`);
  return response.data;
};

export const getAnalytics = async () => {
  const response = await api.get('/analytics');
  return response.data;
};

export default api;
