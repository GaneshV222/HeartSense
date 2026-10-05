import { Upload } from 'lucide-react';

export default function ReportUpload() {
  return (
    <div className="card" style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <h2 style={{ marginTop: 0, color: '#1a365d' }}>Upload Hospital Report</h2>
      <p style={{ color: '#718096', marginBottom: '1.5rem' }}>Upload your medical report and we will extract the required information</p>
      
      <div style={{ 
        border: '2px dashed #38a169', 
        borderRadius: '12px', 
        padding: '3rem 2rem', 
        textAlign: 'center', 
        backgroundColor: '#f0fff4',
        flex: 1,
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'center',
        alignItems: 'center'
      }}>
        <Upload size={48} color="#38a169" style={{ marginBottom: '1rem' }} />
        <p style={{ margin: 0, fontWeight: 600, color: '#276749', fontSize: '1.1rem' }}>Drag & drop your report here</p>
        <p style={{ margin: '0.5rem 0 1.5rem', fontSize: '0.9rem', color: '#2f855a' }}>Supports PDF, JPG, PNG</p>
        
        <button className="btn-primary" style={{ width: 'auto', backgroundColor: '#38a169', padding: '0.75rem 2rem' }}>
          Choose File
        </button>
      </div>
      
      <div style={{ marginTop: '1.5rem' }}>
        <h4 style={{ margin: '0 0 0.75rem 0', color: '#4a5568' }}>SUPPORTED REPORTS</h4>
        <ul style={{ listStyleType: 'none', padding: 0, margin: 0, color: '#718096', lineHeight: '1.8' }}>
          <li>✓ Blood test reports</li>
          <li>✓ Hospital summaries</li>
          <li>✓ Lab reports</li>
          <li>✓ Scanned medical documents</li>
        </ul>
      </div>

      <div style={{ marginTop: '1.5rem', backgroundColor: '#fffaf0', padding: '1rem', borderRadius: '8px', color: '#dd6b20', fontSize: '0.9rem', borderLeft: '4px solid #dd6b20' }}>
        <strong>Current Limitation:</strong> Automatic extraction is not yet fully implemented. Please verify and manually enter values on the left after uploading.
      </div>
    </div>
  );
}
