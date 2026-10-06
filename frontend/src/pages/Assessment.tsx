import { useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import {
  Activity,
  Calendar,
  Clock,
  TrendingUp,
  TrendingDown,
  Minus,
  CheckCircle2,
  AlertTriangle,
  User,
  Search,
  PlusCircle,
  ArrowRight,
  Heart,
} from 'lucide-react';
import AssessmentForm from '../components/AssessmentForm';
import { getPatientTemporalProfile } from '../services/api';

const normalizePatientId = (value: string) => {
  if (!value) return '';
  return String(value)
    .trim()
    .replace(/^patient\s*id\s*[:#-]?\s*/i, '')
    .replace(/^id\s*[:#-]?\s*/i, '')
    .trim();
};

export default function Assessment() {
  const location = useLocation();
  const navigate = useNavigate();
  const [activeTab, setActiveTab] = useState<'existing' | 'new'>('existing');
  const [patientId, setPatientId] = useState('');
  const [lookupError, setLookupError] = useState('');
  const [loading, setLoading] = useState(false);

  const result = location.state?.result;

  const lookupPatient = async (e: any, idToSearch?: string) => {
    if (e) e.preventDefault();
    const rawId = (idToSearch ?? patientId).toString();
    const id = normalizePatientId(rawId);
    if (!id) {
      setLookupError('Please enter a valid Patient ID before searching.');
      return;
    }
    setLookupError('');
    setLoading(true);
    try {
      const data = await getPatientTemporalProfile(id);
      setPatientId(id);
      navigate('/assessment', { state: { result: data } });
    } catch (err: any) {
      setLookupError(err.response?.data?.detail || `Patient "${id}" not found in database.`);
    } finally {
      setLoading(false);
    }
  };

  if (!result) {
    return (
      <div style={{ maxWidth: '1100px', margin: '0 auto' }}>
        <div style={{ marginBottom: '2rem' }}>
          <h1 className="page-title" style={{ fontSize: '2.2rem', color: '#1a365d', marginBottom: '0.5rem' }}>
            Dynamic Cardiovascular Risk Assessment
          </h1>
          <p className="page-subtitle" style={{ color: '#4a5568', fontSize: '1.1rem' }}>
            Predict cardiovascular risk using <strong>current clinical measurements + previous visits + temporal trends + deltas + time metrics</strong> stored in PostgreSQL.
          </p>
        </div>

        {/* Tab Switcher */}
        <div style={{ display: 'flex', gap: '1rem', marginBottom: '2rem', borderBottom: '2px solid #e2e8f0', paddingBottom: '0.5rem' }}>
          <button
            onClick={() => setActiveTab('existing')}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '0.5rem',
              padding: '0.75rem 1.5rem',
              borderRadius: '8px',
              border: 'none',
              cursor: 'pointer',
              fontWeight: 600,
              fontSize: '1rem',
              backgroundColor: activeTab === 'existing' ? '#2b6cb0' : '#edf2f7',
              color: activeTab === 'existing' ? '#ffffff' : '#4a5568',
              transition: 'all 0.2s',
            }}
          >
            <Search size={18} />
            Lookup Existing Patient (Workflow A)
          </button>
          <button
            onClick={() => setActiveTab('new')}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '0.5rem',
              padding: '0.75rem 1.5rem',
              borderRadius: '8px',
              border: 'none',
              cursor: 'pointer',
              fontWeight: 600,
              fontSize: '1rem',
              backgroundColor: activeTab === 'new' ? '#2b6cb0' : '#edf2f7',
              color: activeTab === 'new' ? '#ffffff' : '#4a5568',
              transition: 'all 0.2s',
            }}
          >
            <PlusCircle size={18} />
            Record New Patient / Visit (Workflow B)
          </button>
        </div>

        {activeTab === 'existing' ? (
          <div className="card" style={{ padding: '2.5rem', backgroundColor: '#ffffff', borderRadius: '12px', boxShadow: '0 4px 12px rgba(0,0,0,0.05)' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '1.25rem' }}>
              <div style={{ backgroundColor: '#ebf8ff', padding: '0.75rem', borderRadius: '10px', color: '#2b6cb0' }}>
                <User size={28} />
              </div>
              <div>
                <h2 style={{ margin: 0, color: '#1a365d', fontSize: '1.4rem' }}>Search Patient by ID</h2>
                <p style={{ margin: 0, color: '#718096', fontSize: '0.95rem' }}>
                  Retrieves all historical visits from PostgreSQL, computes chronological features, and predicts using the complete temporal vector.
                </p>
              </div>
            </div>

            <form onSubmit={lookupPatient} style={{ marginTop: '1.5rem' }}>
              <div
                style={{
                  display: 'grid',
                  gridTemplateColumns: '120px minmax(0, 1fr) 220px',
                  alignItems: 'center',
                  gap: '1rem',
                  padding: '1.75rem 1.5rem 1rem',
                  border: '1px solid #dfeaf5',
                  borderRadius: '14px',
                  background: '#f8fbff',
                }}
              >
                <div style={{ fontSize: '1.05rem', fontWeight: 700, color: '#1a365d', lineHeight: 1.2 }}>
                  Patient
                  <br />
                  ID
                </div>

                <div style={{ position: 'relative' }}>
                  <Search
                    size={18}
                    style={{ position: 'absolute', left: '1rem', top: '50%', transform: 'translateY(-50%)', color: '#64748b' }}
                  />
                  <input
                    className="input-field"
                    style={{
                      fontSize: '1.1rem',
                      padding: '0.95rem 1.1rem 0.95rem 2.8rem',
                      width: '100%',
                      border: '1px solid #cbd5e0',
                      borderRadius: '10px',
                      background: '#ffffff',
                      boxShadow: 'inset 0 1px 2px rgba(15, 23, 42, 0.04)',
                    }}
                    value={patientId}
                    onChange={(e) => setPatientId(e.target.value)}
                    placeholder="Enter patient ID (e.g. 45263)"
                    aria-label="Patient ID"
                    autoComplete="off"
                    required
                  />
                </div>

                <button
                  type="submit"
                  className="btn-primary"
                  style={{
                    padding: '0.95rem 1.2rem',
                    fontSize: '1.05rem',
                    whiteSpace: 'nowrap',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    gap: '0.5rem',
                    borderRadius: '10px',
                    boxShadow: '0 8px 20px rgba(59, 130, 246, 0.18)',
                  }}
                  disabled={loading}
                >
                  <Search size={18} />
                  {loading ? 'Analyzing...' : 'Fetch & Predict'}
                </button>
              </div>

              {lookupError && (
                <div style={{ color: '#c53030', backgroundColor: '#fed7d7', padding: '0.75rem 1rem', borderRadius: '6px', marginTop: '0.75rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <AlertTriangle size={18} />
                  <span>{lookupError}</span>
                </div>
              )}
            </form>

            <div style={{ marginTop: '1.5rem', padding: '1rem 1.25rem', backgroundColor: '#f7fafc', borderRadius: '8px', border: '1px solid #e2e8f0' }}>
              <span style={{ display: 'block', fontSize: '0.9rem', color: '#718096', fontWeight: 700, marginBottom: '0.75rem' }}>
                Sample Patient IDs from Ingested Dataset:
              </span>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.6rem' }}>
                {['45263', '49851', '30395', '98758', '95571', '10901', '194', '98910'].map((sampleId) => (
                  <button
                    key={sampleId}
                    type="button"
                    onClick={(e) => {
                      const normalized = normalizePatientId(sampleId);
                      setPatientId(normalized);
                      lookupPatient(e, normalized);
                    }}
                    style={{
                      padding: '0.5rem 0.8rem',
                      backgroundColor: '#ebf8ff',
                      color: '#2b6cb0',
                      border: '1px solid #bee3f8',
                      borderRadius: '999px',
                      cursor: 'pointer',
                      fontSize: '0.875rem',
                      fontWeight: 600,
                      transition: 'all 0.2s ease',
                    }}
                  >
                    ID #{sampleId}
                  </button>
                ))}
              </div>
            </div>
          </div>
        ) : (
          <AssessmentForm />
        )}
      </div>
    );
  }

  // Rendering Patient Temporal Profile
  const predictionValue = result.risk_prediction ?? result.prediction?.prediction ?? 0;
  const probability = result.risk_probability ?? result.prediction?.probability ?? 0;
  const temporal = result.temporal_features || {};
  const current = result.current_values || {};
  const previous = result.previous_values || {};
  const trendInfo = result.trend_information || {};
  const timeline = result.timeline || [];
  const isHighRisk = predictionValue === 1;

  const getTrendBadge = (_trendKey: string, labelKey: string) => {
    const label = trendInfo[labelKey] || temporal[labelKey] || 'Stable';
    if (label === 'Increasing') {
      return (
        <span style={{ display: 'inline-flex', alignItems: 'center', gap: '0.3rem', padding: '0.25rem 0.6rem', borderRadius: '6px', backgroundColor: '#fed7d7', color: '#c53030', fontWeight: 600, fontSize: '0.85rem' }}>
          <TrendingUp size={16} /> Increasing
        </span>
      );
    }
    if (label === 'Decreasing') {
      return (
        <span style={{ display: 'inline-flex', alignItems: 'center', gap: '0.3rem', padding: '0.25rem 0.6rem', borderRadius: '6px', backgroundColor: '#c6f6d5', color: '#22543d', fontWeight: 600, fontSize: '0.85rem' }}>
          <TrendingDown size={16} /> Decreasing
        </span>
      );
    }
    return (
      <span style={{ display: 'inline-flex', alignItems: 'center', gap: '0.3rem', padding: '0.25rem 0.6rem', borderRadius: '6px', backgroundColor: '#edf2f7', color: '#4a5568', fontWeight: 600, fontSize: '0.85rem' }}>
        <Minus size={16} /> {label}
      </span>
    );
  };

  return (
    <div style={{ maxWidth: '1200px', margin: '0 auto' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '2rem' }}>
        <div>
          <h1 className="page-title" style={{ fontSize: '2.2rem', color: '#1a365d', margin: 0 }}>
            Patient Temporal Profile & Risk Analysis
          </h1>
          <p style={{ color: '#718096', margin: '0.25rem 0 0 0', fontSize: '1.05rem' }}>
            Patient ID: <strong>{result.patient_id || result.patient_code}</strong> | Total Visits: <strong>{result.number_of_visits}</strong>
          </p>
        </div>
        <button
          className="btn-primary"
          style={{ width: 'auto', padding: '0.75rem 1.5rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}
          onClick={() => navigate('/assessment', { replace: true, state: {} })}
        >
          <Search size={18} /> New Search / Assessment
        </button>
      </div>

      {/* Main Risk Prediction Banner */}
      <div
        className="card"
        style={{
          marginBottom: '2rem',
          backgroundColor: isHighRisk ? '#fff5f5' : '#f0fff4',
          borderLeft: isHighRisk ? '8px solid #e53e3e' : '8px solid #38a169',
          borderTop: '1px solid #e2e8f0',
          borderRight: '1px solid #e2e8f0',
          borderBottom: '1px solid #e2e8f0',
          padding: '2rem',
          borderRadius: '12px',
        }}
      >
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1.5rem' }}>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
              {isHighRisk ? <AlertTriangle size={32} color="#e53e3e" /> : <CheckCircle2 size={32} color="#38a169" />}
              <h2 style={{ margin: 0, fontSize: '2rem', color: isHighRisk ? '#c53030' : '#22543d' }}>
                {isHighRisk ? 'Higher Cardiovascular Disease Risk' : 'Lower Cardiovascular Disease Risk'}
              </h2>
            </div>
            <p style={{ margin: '0.5rem 0 0 0', color: '#4a5568', fontSize: '1.05rem' }}>
              Assessment calculated using <strong>temporal multi-visit vector + non-linear clinical trends</strong>.
            </p>
          </div>

          <div style={{ display: 'flex', gap: '2rem', alignItems: 'center' }}>
            <div style={{ textAlign: 'center', backgroundColor: '#ffffff', padding: '1rem 1.5rem', borderRadius: '10px', boxShadow: '0 2px 4px rgba(0,0,0,0.05)' }}>
              <div style={{ fontSize: '0.8rem', color: '#718096', textTransform: 'uppercase', fontWeight: 600 }}>Risk Probability</div>
              <div style={{ fontSize: '2.2rem', fontWeight: 800, color: isHighRisk ? '#e53e3e' : '#38a169' }}>
                {(probability * 100).toFixed(1)}%
              </div>
            </div>
            <div style={{ textAlign: 'center', backgroundColor: '#ffffff', padding: '1rem 1.5rem', borderRadius: '10px', boxShadow: '0 2px 4px rgba(0,0,0,0.05)' }}>
              <div style={{ fontSize: '0.8rem', color: '#718096', textTransform: 'uppercase', fontWeight: 600 }}>Current Visit</div>
              <div style={{ fontSize: '2.2rem', fontWeight: 800, color: '#1a365d' }}>
                #{result.visit_number}
              </div>
            </div>
          </div>
        </div>

        {/* Probability Progress Bar */}
        <div style={{ marginTop: '1.5rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', fontWeight: 600, color: '#718096', marginBottom: '0.4rem' }}>
            <span>Low Risk (0%)</span>
            <span>Probability Score: {(probability * 100).toFixed(1)}%</span>
            <span>High Risk (100%)</span>
          </div>
          <div style={{ width: '100%', height: '12px', backgroundColor: '#e2e8f0', borderRadius: '9999px', overflow: 'hidden' }}>
            <div
              style={{
                width: `${Math.min(100, Math.max(5, probability * 100))}%`,
                height: '100%',
                backgroundColor: isHighRisk ? '#e53e3e' : '#38a169',
                borderRadius: '9999px',
                transition: 'width 0.8s ease-in-out',
              }}
            />
          </div>
        </div>
      </div>

      {/* Time & History Overview Metrics */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '1.25rem', marginBottom: '2rem' }}>
        <MetricCard icon={<Clock size={22} color="#3182ce" />} label="Days Since Previous Visit" value={temporal.days_since_previous_visit ?? 0} unit="days" />
        <MetricCard icon={<Calendar size={22} color="#805ad5" />} label="Months Since Previous Visit" value={temporal.months_since_previous_visit ?? 0} unit="months" />
        <MetricCard icon={<Activity size={22} color="#dd6b20" />} label="Prior Recorded Visits" value={temporal.number_of_previous_visits ?? 0} unit="visits" />
        <MetricCard icon={<Heart size={22} color="#e53e3e" />} label="Total Longitudinal Followups" value={result.number_of_visits} unit="total" />
      </div>

      {/* Visit Timeline Section */}
      <div className="card" style={{ marginBottom: '2rem', padding: '2rem', borderRadius: '12px', boxShadow: '0 4px 12px rgba(0,0,0,0.05)' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.25rem', borderBottom: '2px solid #edf2f7', paddingBottom: '0.75rem' }}>
          <div>
            <h3 style={{ margin: 0, color: '#1a365d', fontSize: '1.3rem' }}>Chronological Visit Timeline</h3>
            <p style={{ margin: 0, color: '#718096', fontSize: '0.9rem' }}>PostgreSQL records sorted by visit date</p>
          </div>
          <span style={{ fontSize: '0.85rem', fontWeight: 600, color: '#2b6cb0', backgroundColor: '#ebf8ff', padding: '0.35rem 0.85rem', borderRadius: '9999px' }}>
            {timeline.length} Visit{timeline.length !== 1 ? 's' : ''} in Record
          </span>
        </div>

        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '0.95rem' }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #e2e8f0', color: '#4a5568', backgroundColor: '#f7fafc' }}>
                <th style={{ padding: '0.75rem 1rem' }}>Visit #</th>
                <th style={{ padding: '0.75rem 1rem' }}>Visit Date</th>
                <th style={{ padding: '0.75rem 1rem' }}>BP (Sys / Dia)</th>
                <th style={{ padding: '0.75rem 1rem' }}>Total Cholesterol</th>
                <th style={{ padding: '0.75rem 1rem' }}>HDL / LDL</th>
                <th style={{ padding: '0.75rem 1rem' }}>Glucose / HbA1c</th>
                <th style={{ padding: '0.75rem 1rem' }}>BMI</th>
                <th style={{ padding: '0.75rem 1rem' }}>Heart Rate</th>
                <th style={{ padding: '0.75rem 1rem' }}>Days Between</th>
              </tr>
            </thead>
            <tbody>
              {timeline.map((v: any, i: number) => (
                <tr key={i} style={{ borderBottom: '1px solid #edf2f7', backgroundColor: i === timeline.length - 1 ? '#ebf8ff' : 'transparent' }}>
                  <td style={{ padding: '0.85rem 1rem', fontWeight: 700 }}>
                    Visit {v.visit_number} {i === timeline.length - 1 && <span style={{ color: '#2b6cb0', fontSize: '0.8rem' }}>(Latest)</span>}
                  </td>
                  <td style={{ padding: '0.85rem 1rem', color: '#4a5568' }}>{String(v.visit_date).slice(0, 10)}</td>
                  <td style={{ padding: '0.85rem 1rem', fontWeight: 600 }}>
                    {v.current_systolic_bp ?? '-'} / {v.current_diastolic_bp ?? '-'} mmHg
                  </td>
                  <td style={{ padding: '0.85rem 1rem' }}>{v.current_cholesterol ?? '-'} mg/dL</td>
                  <td style={{ padding: '0.85rem 1rem' }}>{v.current_hdl ?? '-'} / {v.current_ldl ?? '-'}</td>
                  <td style={{ padding: '0.85rem 1rem' }}>
                    {v.current_fasting_blood_sugar ?? '-'} (HbA1c: {v.current_hba1c ?? '-'}%)
                  </td>
                  <td style={{ padding: '0.85rem 1rem' }}>{v.current_bmi ?? '-'}</td>
                  <td style={{ padding: '0.85rem 1rem' }}>{v.current_resting_heart_rate ?? v.current_max_heart_rate ?? '-'} bpm</td>
                  <td style={{ padding: '0.85rem 1rem', color: '#718096' }}>{v.days_since_previous_visit ?? 0} d</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Changes from Previous Visit & Trends Grid */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2rem', marginBottom: '2rem' }}>
        {/* Left: Deltas (Changes from previous visit) */}
        <div className="card" style={{ padding: '2rem', borderRadius: '12px', boxShadow: '0 4px 12px rgba(0,0,0,0.05)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1.25rem', borderBottom: '2px solid #edf2f7', paddingBottom: '0.75rem' }}>
            <Activity size={22} color="#2b6cb0" />
            <h3 style={{ margin: 0, color: '#1a365d', fontSize: '1.25rem' }}>Changes From Immediately Previous Visit (Δ)</h3>
          </div>

          {result.is_first_visit ? (
            <div style={{ padding: '1.5rem', backgroundColor: '#f7fafc', borderRadius: '8px', textAlign: 'center', color: '#718096' }}>
              <p style={{ margin: 0, fontWeight: 500 }}>
                This is the patient's first recorded visit. Baseline clinical values are active and delta variables default safely to zero without loss of predictive power.
              </p>
            </div>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.85rem' }}>
              <DeltaRow label="Δ Systolic Blood Pressure" current={current.systolic_bp} previous={previous.systolic_bp} delta={temporal.delta_systolic_bp} unit="mmHg" />
              <DeltaRow label="Δ Diastolic Blood Pressure" current={current.diastolic_bp} previous={previous.diastolic_bp} delta={temporal.delta_diastolic_bp} unit="mmHg" />
              <DeltaRow label="Δ Total Cholesterol" current={current.cholesterol} previous={previous.cholesterol} delta={temporal.delta_cholesterol} unit="mg/dL" />
              <DeltaRow label="Δ Body Mass Index (BMI)" current={current.bmi} previous={previous.bmi} delta={temporal.delta_bmi} unit="kg/m²" />
              <DeltaRow label="Δ HbA1c / Blood Sugar" current={current.hba1c} previous={previous.hba1c} delta={temporal.delta_hba1c} unit="%" />
              <DeltaRow label="Δ Resting Heart Rate" current={current.resting_heart_rate} previous={previous.resting_heart_rate} delta={temporal.delta_resting_heart_rate} unit="bpm" />
            </div>
          )}
        </div>

        {/* Right: Trends & Lifestyle Trajectory */}
        <div className="card" style={{ padding: '2rem', borderRadius: '12px', boxShadow: '0 4px 12px rgba(0,0,0,0.05)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1.25rem', borderBottom: '2px solid #edf2f7', paddingBottom: '0.75rem' }}>
            <TrendingUp size={22} color="#805ad5" />
            <h3 style={{ margin: 0, color: '#1a365d', fontSize: '1.25rem' }}>Historical Clinical Trends</h3>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
            <TrendRow label="Blood Pressure Trend" badge={getTrendBadge('systolic_bp_trend', 'systolic_bp_trend_label')} desc="Longitudinal systolic BP trajectory across visits" />
            <TrendRow label="Cholesterol Trend" badge={getTrendBadge('cholesterol_trend', 'cholesterol_trend_label')} desc="Serum cholesterol drift across patient followups" />
            <TrendRow label="BMI / Weight Trend" badge={getTrendBadge('bmi_trend', 'bmi_trend_label')} desc="Weight and body mass index rate of change" />
            <TrendRow label="Blood Sugar / HbA1c Trend" badge={getTrendBadge('hba1c_trend', 'hba1c_trend_label')} desc="Glycemic control trajectory" />
            <TrendRow label="Resting Heart Rate Trend" badge={getTrendBadge('resting_heart_rate_trend', 'resting_heart_rate_trend_label')} desc="Cardiovascular resting pulse stability" />
          </div>

          {/* Lifestyle Changes */}
          <div style={{ marginTop: '1.5rem', paddingTop: '1.25rem', borderTop: '1px solid #e2e8f0' }}>
            <h4 style={{ margin: '0 0 0.75rem 0', color: '#2d3748', fontSize: '1rem' }}>Lifestyle & Behavioral Shifts</h4>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.75rem', fontSize: '0.9rem' }}>
              <div style={{ backgroundColor: '#f7fafc', padding: '0.75rem', borderRadius: '6px' }}>
                <span style={{ color: '#718096' }}>Smoking Status Change:</span><br />
                <strong>{temporal.smoking_changed === 1 ? '⚠️ Status Changed' : 'Unchanged'}</strong>
              </div>
              <div style={{ backgroundColor: '#f7fafc', padding: '0.75rem', borderRadius: '6px' }}>
                <span style={{ color: '#718096' }}>Physical Activity Change:</span><br />
                <strong>{temporal.physical_activity_changed === 1 ? '⚠️ Level Changed' : 'Unchanged'}</strong>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Action Footer */}
      <div style={{ display: 'flex', justifyContent: 'center', gap: '1rem', marginTop: '2rem', paddingBottom: '3rem' }}>
        <button
          className="btn-primary"
          style={{ padding: '0.9rem 2rem', fontSize: '1.05rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}
          onClick={() => navigate('/assessment', { replace: true, state: {} })}
        >
          <User size={18} /> Assess Another Patient
        </button>
        <button
          className="btn-primary"
          style={{ padding: '0.9rem 2rem', fontSize: '1.05rem', backgroundColor: '#3182ce', display: 'flex', alignItems: 'center', gap: '0.5rem' }}
          onClick={() => {
            navigate('/analytics');
          }}
        >
          View Model Analytics & Metrics <ArrowRight size={18} />
        </button>
      </div>
    </div>
  );
}

function MetricCard({ icon, label, value, unit }: any) {
  return (
    <div className="card" style={{ padding: '1.25rem', borderRadius: '10px', display: 'flex', alignItems: 'center', gap: '1rem', boxShadow: '0 2px 6px rgba(0,0,0,0.04)' }}>
      <div style={{ backgroundColor: '#f7fafc', padding: '0.75rem', borderRadius: '8px' }}>{icon}</div>
      <div>
        <div style={{ fontSize: '0.8rem', color: '#718096', textTransform: 'uppercase', fontWeight: 600 }}>{label}</div>
        <div style={{ fontSize: '1.4rem', fontWeight: 800, color: '#1a365d' }}>
          {value} <span style={{ fontSize: '0.85rem', fontWeight: 500, color: '#718096' }}>{unit}</span>
        </div>
      </div>
    </div>
  );
}

function DeltaRow({ label, current, previous, delta, unit }: any) {
  const numDelta = Number(delta) || 0;
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '0.65rem 0.85rem', backgroundColor: '#f7fafc', borderRadius: '6px' }}>
      <div>
        <div style={{ fontWeight: 600, color: '#2d3748', fontSize: '0.92rem' }}>{label}</div>
        <div style={{ fontSize: '0.8rem', color: '#718096' }}>
          Prev: {previous ?? '-'} → Curr: {current ?? '-'} {unit}
        </div>
      </div>
      <div style={{ textAlign: 'right' }}>
        <span style={{ fontWeight: 700, fontSize: '1rem', color: numDelta > 0 ? '#e53e3e' : (numDelta < 0 ? '#38a169' : '#718096') }}>
          {numDelta > 0 ? `+${numDelta.toFixed(2)}` : numDelta.toFixed(2)} {unit}
        </span>
      </div>
    </div>
  );
}

function TrendRow({ label, badge, desc }: any) {
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '0.75rem', backgroundColor: '#f7fafc', borderRadius: '6px' }}>
      <div>
        <div style={{ fontWeight: 600, color: '#2d3748', fontSize: '0.95rem' }}>{label}</div>
        <div style={{ fontSize: '0.8rem', color: '#718096' }}>{desc}</div>
      </div>
      <div>{badge}</div>
    </div>
  );
}
