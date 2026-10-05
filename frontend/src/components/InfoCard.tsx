import { Heart } from 'lucide-react';

export default function InfoCard() {
  return (
    <div style={{
      backgroundColor: '#ebf4ff',
      borderRadius: '12px',
      padding: '1.5rem',
      display: 'flex',
      alignItems: 'center',
      gap: '1rem',
      borderLeft: '4px solid #2176ff',
      height: '100%'
    }}>
      <div style={{ backgroundColor: '#2176ff', padding: '0.75rem', borderRadius: '50%' }}>
        <Heart size={24} color="white" />
      </div>
      <div>
        <h3 style={{ margin: '0 0 0.25rem 0', color: '#1a365d', fontSize: '1.1rem' }}>A simple step towards a healthier tomorrow</h3>
        <p style={{ margin: 0, color: '#4a5568', fontSize: '0.95rem' }}>Know your risk. Take control.</p>
      </div>
    </div>
  );
}
