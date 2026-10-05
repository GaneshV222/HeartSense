import { useNavigate } from 'react-router-dom';
import InfoCard from '../components/InfoCard';
import FeatureCards from '../components/FeatureCards';
import { Activity } from 'lucide-react';

export default function Home() {
  const navigate = useNavigate();
  return (
    <div>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 300px', gap: '2rem', marginBottom: '3rem' }}>
        <div>
          <h1 className="page-title">Cardiovascular Health Assessment</h1>
          <p className="page-subtitle" style={{ marginBottom: '1.5rem', fontSize: '1.15rem', color: '#4a5568' }}>
            Understand your cardiovascular risk using your clinical information and our machine-learning based assessment system.
          </p>
          <div style={{ backgroundColor: '#fff', padding: '2rem', borderRadius: '12px', border: '1px solid #e2e8f0', boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.05)' }}>
            <h2 style={{ marginTop: 0, color: '#1a365d', marginBottom: '1rem' }}>Know Your Cardiovascular Risk</h2>
            <p style={{ color: '#4a5568', lineHeight: '1.6', marginBottom: '2rem' }}>
              Our system evaluates your clinical information—such as blood pressure, cholesterol levels, and age—against a robust dataset to provide a comprehensive cardiovascular risk assessment. 
            </p>
            <button className="btn-primary" style={{ padding: '1rem 2.5rem', fontSize: '1.1rem', display: 'inline-flex', alignItems: 'center', gap: '0.75rem' }} onClick={() => navigate('/assessment')}>
              <Activity size={24} />
              Start Assessment
            </button>
          </div>
        </div>
        <InfoCard />
      </div>

      <FeatureCards />
    </div>
  );
}
