import { NavLink } from 'react-router-dom';
import { HeartPulse, Home, Search, PlusCircle, BarChart2, Info } from 'lucide-react';

export default function Sidebar() {
  const linkStyle = {
    display: 'flex',
    alignItems: 'center',
    padding: '0.75rem 1rem',
    color: '#4a5568',
    textDecoration: 'none',
    borderRadius: '8px',
    marginBottom: '0.5rem',
    fontWeight: 500,
    transition: 'all 0.2s',
  };
  const activeStyle = {
    ...linkStyle,
    backgroundColor: '#ebf4ff',
    color: '#2176ff',
    fontWeight: 600,
  };

  return (
    <div style={{ width: '260px', backgroundColor: '#ffffff', borderRight: '1px solid #e2e8f0', padding: '1.5rem', display: 'flex', flexDirection: 'column' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '2rem' }}>
        <HeartPulse size={28} color="#e53e3e" />
        <div>
          <h2 style={{ margin: 0, color: '#1a365d', fontSize: '1.25rem' }}>HeartSense</h2>
          <span style={{ fontSize: '0.8rem', color: '#718096' }}>Temporal CVD Risk</span>
        </div>
      </div>
      <nav style={{ flex: 1 }}>
        <NavLink to="/" style={({ isActive }) => isActive ? activeStyle : linkStyle}>
          <Home size={20} style={{ marginRight: '0.75rem' }} /> Home
        </NavLink>
        <NavLink to="/assessment" style={({ isActive }) => isActive ? activeStyle : linkStyle}>
          <Search size={20} style={{ marginRight: '0.75rem' }} /> Lookup Patient
        </NavLink>
        <NavLink to="/new-assessment" style={({ isActive }) => isActive ? activeStyle : linkStyle}>
          <PlusCircle size={20} style={{ marginRight: '0.75rem' }} /> Record New Patient
        </NavLink>
        <NavLink to="/analytics" style={({ isActive }) => isActive ? activeStyle : linkStyle}>
          <BarChart2 size={20} style={{ marginRight: '0.75rem' }} /> Analytics
        </NavLink>
        <NavLink to="/about" style={({ isActive }) => isActive ? activeStyle : linkStyle}>
          <Info size={20} style={{ marginRight: '0.75rem' }} /> About
        </NavLink>
      </nav>
    </div>
  );
}
