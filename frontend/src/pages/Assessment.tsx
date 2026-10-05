import { useLocation, useNavigate } from 'react-router-dom';
import AssessmentForm from '../components/AssessmentForm';
import ReportUpload from '../components/ReportUpload';

export default function Assessment() {
  const location = useLocation();
  const navigate = useNavigate();
  const result = location.state?.result;

  if (!result) {
    return (
      <div>
        <h1 className="page-title">Cardiovascular Health Assessment</h1>
        <p className="page-subtitle" style={{ marginBottom: '2rem' }}>
          Enter your health information or upload your medical report to assess your current cardiovascular risk.
        </p>
        
        <div style={{ display: 'grid', gridTemplateColumns: '3fr 2fr', gap: '2rem' }}>
          <AssessmentForm />
          <ReportUpload />
        </div>
      </div>
    );
  }

  const { prediction, patient_code, patient_name, is_first_visit, temporal_changes } = result;

  return (
    <div>
      <h1 className="page-title">Latest Assessment</h1>
      <p className="page-subtitle">Assessment generated on {new Date().toLocaleDateString()} for {patient_code} {patient_name ? `(${patient_name})` : ''}</p>
      
      <div className="card" style={{ marginBottom: '2rem', borderLeft: prediction.prediction === 1 ? '6px solid #e53e3e' : '6px solid #38a169', padding: '2rem' }}>
        <h2 style={{ marginTop: 0, fontSize: '2rem', color: prediction.prediction === 1 ? '#e53e3e' : '#38a169' }}>
          {prediction.prediction === 1 ? '🔴 Disease Predicted (Higher Risk)' : '🟢 No Disease Predicted (Lower Risk)'}
        </h2>
        
        <div style={{ display: 'flex', gap: '4rem', marginTop: '2rem' }}>
          <div>
            <p style={{ margin: '0 0 0.5rem 0', color: '#718096', textTransform: 'uppercase', fontSize: '0.875rem', fontWeight: 600 }}>Prediction Probability</p>
            <p style={{ margin: 0, fontSize: '2.5rem', fontWeight: 'bold', color: '#1a365d' }}>{(prediction.probability * 100).toFixed(1)}%</p>
          </div>
          <div>
            <p style={{ margin: '0 0 0.5rem 0', color: '#718096', textTransform: 'uppercase', fontSize: '0.875rem', fontWeight: 600 }}>Prediction Model</p>
            <p style={{ margin: 0, fontSize: '2.5rem', fontWeight: 'bold', color: '#1a365d' }}>{prediction.model_name}</p>
          </div>
        </div>
        
        <div style={{ marginTop: '2rem', padding: '1rem', backgroundColor: '#fffaf0', borderLeft: '4px solid #dd6b20', borderRadius: '4px' }}>
          <p style={{ margin: 0, color: '#9c4221', fontSize: '0.95rem' }}>
            <strong>Disclaimer:</strong> This assessment is for research/decision-support purposes and is not a medical diagnosis. Consult a qualified healthcare professional for medical advice.
          </p>
        </div>
      </div>

      {!is_first_visit && temporal_changes && temporal_changes.length > 0 ? (
        <div className="card">
          <h3 style={{ margin: '0 0 1.5rem 0', color: '#1a365d' }}>Temporal Changes Since Last Visit</h3>
          <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left' }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #e2e8f0', color: '#718096' }}>
                <th style={{ padding: '0.75rem' }}>Parameter</th>
                <th style={{ padding: '0.75rem' }}>Previous</th>
                <th style={{ padding: '0.75rem' }}>Current</th>
                <th style={{ padding: '0.75rem' }}>Change</th>
                <th style={{ padding: '0.75rem' }}>Status</th>
              </tr>
            </thead>
            <tbody>
              {temporal_changes.map((change: any, idx: number) => (
                <tr key={idx} style={{ borderBottom: '1px solid #edf2f7' }}>
                  <td style={{ padding: '0.75rem', fontWeight: 500 }}>{change.label.split(' (')[0]}</td>
                  <td style={{ padding: '0.75rem' }}>{change.previous}</td>
                  <td style={{ padding: '0.75rem', fontWeight: 'bold' }}>{change.current}</td>
                  <td style={{ padding: '0.75rem' }}>{(Number(change.current) - Number(change.previous)).toFixed(1)}</td>
                  <td style={{ padding: '0.75rem' }}>
                    <span style={{
                      backgroundColor: change.direction === 'Increased' ? '#fed7d7' : change.direction === 'Decreased' ? '#c6f6d5' : '#edf2f7',
                      color: change.direction === 'Increased' ? '#c53030' : change.direction === 'Decreased' ? '#2f855a' : '#4a5568',
                      padding: '0.25rem 0.75rem',
                      borderRadius: '9999px',
                      fontSize: '0.875rem',
                      fontWeight: 500
                    }}>
                      {change.direction}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        !is_first_visit && (
          <div className="card">
            <h3 style={{ margin: '0 0 1rem 0', color: '#1a365d' }}>Temporal Changes</h3>
            <p style={{ color: '#718096' }}>Not enough historical data for temporal comparison.</p>
          </div>
        )
      )}
      
      <div style={{ marginTop: '2rem', textAlign: 'center' }}>
        <button className="btn-primary" style={{ width: 'auto' }} onClick={() => navigate('/', { replace: true })}>
          Return Home
        </button>
      </div>
    </div>
  );
}
