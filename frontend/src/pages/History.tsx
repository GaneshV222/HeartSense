import { useState } from 'react';
import { getHistory } from '../services/api';
import { Search } from 'lucide-react';

export default function History() {
  const [patientCode, setPatientCode] = useState('');
  const [history, setHistory] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const handleSearch = async (e: any) => {
    e.preventDefault();
    if (!patientCode) return;
    
    setLoading(true);
    setError(null);
    try {
      const data = await getHistory(patientCode);
      setHistory(data);
    } catch (err: any) {
      setError(err.response?.data?.detail || "Patient not found or error loading history.");
      setHistory(null);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <h1 className="page-title">My History</h1>
      <p className="page-subtitle">View your previous health assessments.</p>
      
      <div className="card" style={{ marginBottom: '2rem' }}>
        <form onSubmit={handleSearch} style={{ display: 'flex', gap: '1rem', alignItems: 'end' }}>
          <div className="form-group" style={{ flex: 1, marginBottom: 0 }}>
            <label>Search Patient Code</label>
            <input 
              type="text" 
              className="input-field" 
              value={patientCode} 
              onChange={e => setPatientCode(e.target.value)} 
              placeholder="e.g. P001" 
            />
          </div>
          <button type="submit" className="btn-primary" style={{ width: 'auto', display: 'flex', alignItems: 'center', gap: '0.5rem' }} disabled={loading}>
            <Search size={18} /> {loading ? 'Searching...' : 'Search'}
          </button>
        </form>
      </div>

      {error && <div style={{ backgroundColor: '#fed7d7', color: '#c53030', padding: '1rem', borderRadius: '8px', marginBottom: '1.5rem' }}>{error}</div>}

      {history && (
        <div>
          <h2 style={{ color: '#1a365d' }}>Patient: {history.patient.patient_code} {history.patient.name ? `(${history.patient.name})` : ''}</h2>
          
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem', marginTop: '1.5rem' }}>
            {history.timeline.map((visit: any, index: number) => (
              <div key={index} className="card" style={{ borderLeft: visit.prediction === 1 ? '4px solid #e53e3e' : '4px solid #38a169' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem', borderBottom: '1px solid #e2e8f0', paddingBottom: '0.5rem' }}>
                  <h3 style={{ margin: 0 }}>Visit {index + 1} - {visit.timestamp ? visit.timestamp.split('T')[0] : 'Unknown Date'}</h3>
                  <div style={{ fontWeight: 'bold', color: visit.prediction === 1 ? '#e53e3e' : '#38a169' }}>
                    {visit.prediction === 1 ? 'Needs Attention' : 'Normal'}
                  </div>
                </div>
                
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '1rem' }}>
                  {Object.entries(visit.features).slice(0, 8).map(([key, value]: any) => (
                    <div key={key}>
                      <span style={{ display: 'block', fontSize: '0.85rem', color: '#718096' }}>{key}</span>
                      <span style={{ fontWeight: 500 }}>{value}</span>
                    </div>
                  ))}
                </div>
                
                {visit.probability !== null && (
                  <div style={{ marginTop: '1rem', paddingTop: '1rem', borderTop: '1px solid #e2e8f0' }}>
                    <span style={{ color: '#718096', marginRight: '0.5rem' }}>Risk Probability:</span>
                    <strong style={{ color: visit.prediction === 1 ? '#e53e3e' : '#38a169' }}>
                      {(visit.probability * 100).toFixed(1)}%
                    </strong>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
