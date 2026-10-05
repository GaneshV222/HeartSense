import { Zap, Activity, Clock, ShieldCheck } from 'lucide-react';

export default function FeatureCards() {
  const cards = [
    {
      title: 'Instant Assessment',
      desc: 'Get your cardiovascular risk prediction in seconds',
      icon: <Zap size={24} color="#3182ce" />,
      bg: '#ebf8ff'
    },
    {
      title: 'Data-Driven Analysis',
      desc: 'Powered by machine learning models',
      icon: <Activity size={24} color="#38a169" />,
      bg: '#f0fff4'
    },
    {
      title: 'Track Your Progress',
      desc: 'View your past assessments and changes over time',
      icon: <Clock size={24} color="#805ad5" />,
      bg: '#faf5ff'
    },
    {
      title: 'Better Health Decisions',
      desc: 'Understand your risk and take timely action',
      icon: <ShieldCheck size={24} color="#e53e3e" />,
      bg: '#fff5f5'
    }
  ];

  return (
    <div style={{
      display: 'grid',
      gridTemplateColumns: 'repeat(4, 1fr)',
      gap: '1.5rem',
      marginTop: '2rem'
    }}>
      {cards.map((card, idx) => (
        <div key={idx} className="card" style={{ backgroundColor: card.bg, border: 'none' }}>
          <div style={{ marginBottom: '1rem' }}>{card.icon}</div>
          <h3 style={{ margin: '0 0 0.5rem 0', color: '#1a365d', fontSize: '1rem' }}>{card.title}</h3>
          <p style={{ margin: 0, color: '#4a5568', fontSize: '0.875rem', lineHeight: '1.4' }}>{card.desc}</p>
        </div>
      ))}
    </div>
  );
}
