import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Activity } from 'lucide-react';
import { assessRisk } from '../services/api';

export default function AssessmentForm() {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [formData, setFormData] = useState({
    patientCode: '',
    patientName: '',
    age: 50,
    sex: 1,
    cp: 0,
    trestbps: 120,
    chol: 200,
    fbs: 0,
    restecg: 0,
    thalach: 150,
    exang: 0,
    oldpeak: 1.0,
    slope: 0,
    ca: 0,
    thal: 0
  });

  const handleChange = (e: any) => {
    const { name, value, type } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: type === 'number' ? Number(value) : value
    }));
  };

  const handleSubmit = async (e: any) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const { patientCode, patientName, ...clinicalData } = formData;
      if (!patientCode) {
        throw new Error("Patient Code is required");
      }
      const data = await assessRisk({ patientCode, patientName, clinicalData });
      navigate('/assessment', { state: { result: data } });
    } catch (err: any) {
      setError(err.response?.data?.detail || err.message || "An error occurred");
      setLoading(false);
    }
  };

  return (
    <div className="card">
      <h2 style={{ marginTop: 0, color: '#1a365d' }}>Enter Your Details</h2>
      <p style={{ color: '#718096', marginBottom: '1.5rem' }}>Fill in your health information below</p>
      
      {error && <div style={{ backgroundColor: '#fed7d7', color: '#c53030', padding: '1rem', borderRadius: '8px', marginBottom: '1.5rem' }}>{error}</div>}

      <form onSubmit={handleSubmit}>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem', marginBottom: '2rem' }}>
          <div className="form-group" style={{ marginBottom: 0 }}>
            <label>Patient Code *</label>
            <input type="text" className="input-field" name="patientCode" value={formData.patientCode} onChange={handleChange} required placeholder="e.g. P001" style={{ padding: '0.85rem' }} />
          </div>
          <div className="form-group" style={{ marginBottom: 0 }}>
            <label>Name</label>
            <input type="text" className="input-field" name="patientName" value={formData.patientName} onChange={handleChange} placeholder="e.g. John Doe" style={{ padding: '0.85rem' }} />
          </div>
        </div>

        <h3 style={{ borderBottom: '1px solid #e2e8f0', paddingBottom: '0.5rem', color: '#2d3748', fontSize: '1.1rem' }}>Clinical Data</h3>
        
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '1.25rem' }}>
          <div className="form-group">
            <label>Age (years)</label>
            <input type="number" className="input-field" name="age" value={formData.age} onChange={handleChange} min={1} max={120} />
          </div>
          <div className="form-group">
            <label>Gender</label>
            <select className="input-field" name="sex" value={formData.sex} onChange={handleChange}>
              <option value={1}>Male</option>
              <option value={0}>Female</option>
            </select>
          </div>
          <div className="form-group">
            <label>Chest Pain Type</label>
            <select className="input-field" name="cp" value={formData.cp} onChange={handleChange}>
              <option value={0}>Typical Angina</option>
              <option value={1}>Atypical Angina</option>
              <option value={2}>Non-anginal Pain</option>
              <option value={3}>Asymptomatic</option>
            </select>
          </div>
          
          <div className="form-group">
            <label>Blood Pressure (Systolic)</label>
            <input type="number" className="input-field" name="trestbps" value={formData.trestbps} onChange={handleChange} min={50} max={250} placeholder="mm Hg" />
          </div>
          <div className="form-group">
            <label>Cholesterol</label>
            <input type="number" className="input-field" name="chol" value={formData.chol} onChange={handleChange} min={50} max={600} placeholder="mg/dl" />
          </div>
          <div className="form-group">
            <label>Fasting Blood Sugar {'>'} 120</label>
            <select className="input-field" name="fbs" value={formData.fbs} onChange={handleChange}>
              <option value={0}>No</option>
              <option value={1}>Yes</option>
            </select>
          </div>
          
          <div className="form-group">
            <label>Resting ECG</label>
            <select className="input-field" name="restecg" value={formData.restecg} onChange={handleChange}>
              <option value={0}>Normal</option>
              <option value={1}>ST-T Abnormality</option>
              <option value={2}>LV Hypertrophy</option>
            </select>
          </div>
          <div className="form-group">
            <label>Maximum Heart Rate</label>
            <input type="number" className="input-field" name="thalach" value={formData.thalach} onChange={handleChange} min={50} max={250} />
          </div>
          <div className="form-group">
            <label>Exercise Induced Angina</label>
            <select className="input-field" name="exang" value={formData.exang} onChange={handleChange}>
              <option value={0}>No</option>
              <option value={1}>Yes</option>
            </select>
          </div>
          
          <div className="form-group">
            <label>ST Depression (oldpeak)</label>
            <input type="number" className="input-field" name="oldpeak" value={formData.oldpeak} onChange={handleChange} min={0} max={10} step={0.1} />
          </div>
          <div className="form-group">
            <label>Slope</label>
            <select className="input-field" name="slope" value={formData.slope} onChange={handleChange}>
              <option value={0}>Upsloping</option>
              <option value={1}>Flat</option>
              <option value={2}>Downsloping</option>
            </select>
          </div>
          <div className="form-group">
            <label>No. of Major Vessels</label>
            <select className="input-field" name="ca" value={formData.ca} onChange={handleChange}>
              <option value={0}>0</option>
              <option value={1}>1</option>
              <option value={2}>2</option>
              <option value={3}>3</option>
            </select>
          </div>
          
          <div className="form-group" style={{ gridColumn: '1 / -1' }}>
            <label>Thalassemia</label>
            <select className="input-field" name="thal" value={formData.thal} onChange={handleChange}>
              <option value={0}>Normal</option>
              <option value={1}>Fixed Defect</option>
              <option value={2}>Reversible Defect</option>
            </select>
          </div>
        </div>

        <button type="submit" className="btn-primary" style={{ marginTop: '1.5rem', padding: '1.25rem', fontSize: '1.15rem', display: 'flex', justifyContent: 'center', alignItems: 'center', gap: '0.75rem' }} disabled={loading}>
          <Activity size={24} /> {loading ? 'Assessing...' : 'Assess My Risk'}
        </button>
      </form>
    </div>
  );
}
