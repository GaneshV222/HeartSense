import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Activity } from 'lucide-react';
import { assessRisk } from '../services/api';

const normalizePatientCode = (value: string) => {
  if (!value) return '';
  return String(value)
    .trim()
    .replace(/^patient\s*id\s*[:#-]?\s*/i, '')
    .replace(/^id\s*[:#-]?\s*/i, '')
    .trim();
};

export default function AssessmentForm() {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [formData, setFormData] = useState({
    patientCode: '', patientName: '', age: 50, gender: 'Male', bmi: 27,
    chest_pain_type: 'Typical Angina', systolic_bp: 120, diastolic_bp: 80,
    resting_heart_rate: 75, max_heart_rate: 150, cholesterol: 200, hdl: 50, ldl: 110,
    fasting_blood_sugar: 'False', hba1c: 5.5, diabetes: 'False', resting_ecg: 'Normal',
    exercise_angina: 'False', oldpeak: 1.0, st_slope: 'Flat', num_major_vessels: 0,
    thalassemia: 'Normal', smoking: 'False', family_history: 'False',
    physical_activity: 'Moderate', stress_level: 5
  });
  const handleChange = (e: any) => {
    const { name, value, type } = e.target;
    const nextValue = name === 'patientCode' ? normalizePatientCode(value) : type === 'number' ? Number(value) : value;
    setFormData(prev => ({ ...prev, [name]: nextValue }));
  };
  const handleSubmit = async (e: any) => {
    e.preventDefault(); setLoading(true); setError(null);
    try {
      const patientCode = normalizePatientCode(formData.patientCode);
      const patientName = formData.patientName?.trim();
      if (!patientCode) throw new Error('Patient ID is required');
      const { patientCode: _ignoredCode, patientName: _ignoredName, ...clinicalData } = formData;
      const data = await assessRisk({ patientCode, patientName, clinicalData });
      navigate('/assessment', { state: { result: data } });
    } catch (err: any) { setError(err.response?.data?.detail || err.message || 'An error occurred'); setLoading(false); }
  };
  return (
    <div className="card">
      <h2 style={{ marginTop: 0, color: '#1a365d' }}>New Patient / New Visit</h2>
      {error && <div style={{ backgroundColor: '#fed7d7', color: '#c53030', padding: '1rem', borderRadius: '8px', marginBottom: '1.5rem' }}>{error}</div>}
      <form onSubmit={handleSubmit}>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem', marginBottom: '2rem' }}>
          <div className="form-group" style={{ marginBottom: 0 }}>
            <label>Patient ID *</label>
            <input
              type="text"
              className="input-field"
              name="patientCode"
              value={formData.patientCode}
              onChange={handleChange}
              required
              placeholder="e.g. P4505 or P001"
              autoComplete="off"
            />
          </div>
          <div className="form-group" style={{ marginBottom: 0 }}><label>Name</label><input type="text" className="input-field" name="patientName" value={formData.patientName} onChange={handleChange} placeholder="optional" /></div>
        </div>
        <h3 style={{ borderBottom: '1px solid #e2e8f0', paddingBottom: '0.5rem', color: '#2d3748', fontSize: '1.1rem' }}>Clinical + Lifestyle Data</h3>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '1.25rem' }}>
          <Field label="Age" name="age" value={formData.age} onChange={handleChange} type="number" />
          <Select label="Gender" name="gender" value={formData.gender} onChange={handleChange} options={['Male','Female']} />
          <Field label="BMI" name="bmi" value={formData.bmi} onChange={handleChange} type="number" step="0.1" />
          <Select label="Chest Pain" name="chest_pain_type" value={formData.chest_pain_type} onChange={handleChange} options={['Typical Angina','Atypical Angina','Non-anginal Pain','Asymptomatic']} />
          <Field label="Systolic BP" name="systolic_bp" value={formData.systolic_bp} onChange={handleChange} type="number" />
          <Field label="Diastolic BP" name="diastolic_bp" value={formData.diastolic_bp} onChange={handleChange} type="number" />
          <Field label="Resting Heart Rate" name="resting_heart_rate" value={formData.resting_heart_rate} onChange={handleChange} type="number" />
          <Field label="Max Heart Rate" name="max_heart_rate" value={formData.max_heart_rate} onChange={handleChange} type="number" />
          <Field label="Cholesterol" name="cholesterol" value={formData.cholesterol} onChange={handleChange} type="number" />
          <Field label="HDL" name="hdl" value={formData.hdl} onChange={handleChange} type="number" />
          <Field label="LDL" name="ldl" value={formData.ldl} onChange={handleChange} type="number" />
          <Select label="Fasting Blood Sugar" name="fasting_blood_sugar" value={formData.fasting_blood_sugar} onChange={handleChange} options={['False','True']} />
          <Field label="HbA1c" name="hba1c" value={formData.hba1c} onChange={handleChange} type="number" step="0.1" />
          <Select label="Diabetes" name="diabetes" value={formData.diabetes} onChange={handleChange} options={['False','True']} />
          <Select label="Resting ECG" name="resting_ecg" value={formData.resting_ecg} onChange={handleChange} options={['Normal','ST-T Wave Abnormality','Left Ventricular Hypertrophy']} />
          <Select label="Exercise Angina" name="exercise_angina" value={formData.exercise_angina} onChange={handleChange} options={['False','True']} />
          <Field label="Oldpeak" name="oldpeak" value={formData.oldpeak} onChange={handleChange} type="number" step="0.1" />
          <Select label="ST Slope" name="st_slope" value={formData.st_slope} onChange={handleChange} options={['Upsloping','Flat','Downsloping']} />
          <Field label="Major Vessels" name="num_major_vessels" value={formData.num_major_vessels} onChange={handleChange} type="number" />
          <Select label="Thalassemia" name="thalassemia" value={formData.thalassemia} onChange={handleChange} options={['Normal','Fixed Defect','Reversible Defect']} />
          <Select label="Smoking" name="smoking" value={formData.smoking} onChange={handleChange} options={['False','True']} />
          <Select label="Family History" name="family_history" value={formData.family_history} onChange={handleChange} options={['False','True']} />
          <Select label="Physical Activity" name="physical_activity" value={formData.physical_activity} onChange={handleChange} options={['Low','Moderate','High']} />
          <Field label="Stress Level" name="stress_level" value={formData.stress_level} onChange={handleChange} type="number" />
        </div>
        <button type="submit" className="btn-primary" style={{ marginTop: '1.5rem', padding: '1.25rem', fontSize: '1.15rem', display: 'flex', justifyContent: 'center', alignItems: 'center', gap: '0.75rem' }} disabled={loading}><Activity size={24} /> {loading ? 'Assessing...' : 'Save Visit & Assess Risk'}</button>
      </form>
    </div>
  );
}
function Field(props: any) { return <div className="form-group"><label>{props.label}</label><input className="input-field" {...props} label={undefined} /></div>; }
function Select({ label, name, value, onChange, options }: any) { return <div className="form-group"><label>{label}</label><select className="input-field" name={name} value={value} onChange={onChange}>{options.map((o: string) => <option key={o} value={o}>{o}</option>)}</select></div>; }
