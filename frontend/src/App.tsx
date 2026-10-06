import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import Sidebar from './components/layout/Sidebar';
import PageContainer from './components/layout/PageContainer';
import Home from './pages/Home';
import Assessment from './pages/Assessment';
import NewAssessment from './pages/NewAssessment';
import Analytics from './pages/Analytics';
import About from './pages/About';
import './App.css';

function App() {
  return (
    <Router>
      <div style={{ display: 'flex', minHeight: '100vh', backgroundColor: '#f8fbff' }}>
        <Sidebar />
        <PageContainer>
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/assessment" element={<Assessment />} />
            <Route path="/new-assessment" element={<NewAssessment />} />
            <Route path="/analytics" element={<Analytics />} />
            <Route path="/about" element={<About />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </PageContainer>
      </div>
    </Router>
  );
}

export default App;
