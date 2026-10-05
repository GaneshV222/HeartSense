export default function About() {
  return (
    <div>
      <h1 className="page-title">About HeartSense</h1>
      <p className="page-subtitle">Cardiovascular Disease Prediction System</p>
      
      <div className="card">
        <h2 style={{ marginTop: 0 }}>Purpose</h2>
        <p>To provide a machine-learning-based cardiovascular risk assessment using patient clinical information.</p>
        
        <h2 style={{ marginTop: '2rem' }}>How it Works</h2>
        <ul style={{ lineHeight: '1.8' }}>
          <li><strong>Advanced Analysis:</strong> The system uses machine learning (XGBoost, Random Forest, etc.) to analyze patient clinical information.</li>
          <li><strong>Historical Tracking:</strong> Patient visits are logged, and historical data can be compared to identify health trends.</li>
        </ul>
        
        <div style={{ marginTop: '2rem', padding: '1.5rem', backgroundColor: '#fffaf0', borderLeft: '4px solid #dd6b20', borderRadius: '0 8px 8px 0' }}>
          <h3 style={{ margin: '0 0 0.5rem 0', color: '#c05621' }}>Important Disclaimer</h3>
          <p style={{ margin: 0, color: '#7b341e' }}>
            The system is intended for research and decision-support purposes only. It is <strong>NOT</strong> a medical diagnosis. Always consult a qualified healthcare professional.
          </p>
        </div>
      </div>
    </div>
  );
}
