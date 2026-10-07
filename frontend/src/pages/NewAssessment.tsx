import AssessmentForm from '../components/AssessmentForm';

export default function NewAssessment() {
  return (
    <div style={{ maxWidth: '1100px', margin: '0 auto', paddingBottom: '3rem' }}>
      <div style={{ marginBottom: '2rem' }}>
        <h1 className="page-title" style={{ fontSize: '2.2rem', color: '#1a365d', marginBottom: '0.5rem' }}>
          Record New Patient Assessment
        </h1>
      </div>

      <AssessmentForm />
    </div>
  );
}
