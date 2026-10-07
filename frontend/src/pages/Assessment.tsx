import { useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import {
  Activity,
  CheckCircle2,
  AlertTriangle,
  User,
  Search,
  PlusCircle,
  ArrowRight,
  Heart,
} from 'lucide-react';
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
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '2rem', flexWrap: 'wrap', gap: '1rem' }}>
          <div>
            <h1 className="page-title" style={{ fontSize: '2.2rem', color: '#1a365d', marginBottom: '0.5rem' }}>
              Lookup Existing Patient
            </h1>
          </div>
          <button
            onClick={() => navigate('/new-assessment')}
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '0.5rem',
              padding: '0.75rem 1.25rem',
              borderRadius: '8px',
              border: '1px solid #bee3f8',
              cursor: 'pointer',
              fontWeight: 600,
              fontSize: '0.95rem',
              backgroundColor: '#ebf8ff',
              color: '#2b6cb0',
              transition: 'all 0.2s',
            }}
          >
            <PlusCircle size={18} />
            Record New Patient / Visit
          </button>
        </div>

        <div className="card" style={{ padding: '2.5rem', backgroundColor: '#ffffff', borderRadius: '12px', boxShadow: '0 4px 12px rgba(0,0,0,0.05)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '1.25rem' }}>
            <div style={{ backgroundColor: '#ebf8ff', padding: '0.75rem', borderRadius: '10px', color: '#2b6cb0' }}>
              <User size={28} />
            </div>
            <div>
              <h2 style={{ margin: 0, color: '#1a365d', fontSize: '1.4rem' }}>Search Patient by ID</h2>
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
              Sample Patient ID
            </span>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.6rem' }}>
              {['P56393', 'P4505', 'P46414', 'P36228', 'P31853', 'P100355'].map((sampleId) => (
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
                  {sampleId}
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>
    );
  }

  // Rendering Patient Temporal Profile
  const predictionValue = result.risk_prediction ?? result.prediction?.prediction ?? 0;
  const isFirstVisit = result.is_first_visit !== undefined ? result.is_first_visit : ((result.number_of_visits || 1) <= 1);
  const snapshot = result.temporal_snapshot || {};
  const temporal = result.temporal_snapshot || result.temporal_features || {};
  const current = result.current_values || snapshot || {};
  const previous = result.previous_values || {};
  const timeline = result.timeline || [];
  const isHighRisk = predictionValue === 1;

  return (
    <div style={{ maxWidth: '1200px', margin: '0 auto', paddingBottom: '3rem' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '2rem' }}>
        <div>
          <h1 className="page-title" style={{ fontSize: '2.2rem', color: '#1a365d', margin: 0 }}>
            Patient Temporal Profile & Risk Analysis
          </h1>
          <p style={{ color: '#718096', margin: '0.25rem 0 0 0', fontSize: '1.05rem' }}>
            Patient ID: <strong>{result.patient_id || result.patient_code}</strong> | Total Visits: <strong>{result.number_of_visits}</strong> | Current Visit: <strong>#{result.visit_number}</strong>
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
          </div>

        </div>

      </div>

      {/* Changes from Immediately Previous Visit Section (Section 2, 3, 25, 26, 27) */}
      <div style={{ display: 'grid', gridTemplateColumns: '1.3fr 0.9fr', gap: '2rem', marginBottom: '2rem' }}>
        {/* Left: 8 Numerical Variables */}
        <div className="card" style={{ padding: '1.75rem', borderRadius: '12px', boxShadow: '0 4px 12px rgba(0,0,0,0.05)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1.25rem', borderBottom: '2px solid #edf2f7', paddingBottom: '0.75rem' }}>
            <Activity size={22} color="#2b6cb0" />
            <h3 style={{ margin: 0, color: '#1a365d', fontSize: '1.25rem' }}>Numerical Clinical Deltas (8 Variables)</h3>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
            <DeltaCard
              label="Systolic BP"
              unit="mmHg"
              curr={current.systolic_bp}
              prev={previous.systolic_bp}
              delta={temporal.delta_systolic_bp}
              isFirst={isFirstVisit}
            />
            <DeltaCard
              label="Diastolic BP"
              unit="mmHg"
              curr={current.diastolic_bp}
              prev={previous.diastolic_bp}
              delta={temporal.delta_diastolic_bp}
              isFirst={isFirstVisit}
            />
            <DeltaCard
              label="Total Cholesterol"
              unit="mg/dL"
              curr={current.cholesterol}
              prev={previous.cholesterol}
              delta={temporal.delta_cholesterol}
              isFirst={isFirstVisit}
            />
            <DeltaCard
              label="LDL Cholesterol"
              unit="mg/dL"
              curr={current.ldl}
              prev={previous.ldl}
              delta={temporal.delta_ldl}
              isFirst={isFirstVisit}
            />
            <DeltaCard
              label="HDL Cholesterol"
              unit="mg/dL"
              curr={current.hdl}
              prev={previous.hdl}
              delta={temporal.delta_hdl}
              isFirst={isFirstVisit}
            />
            <DeltaCard
              label="BMI"
              unit="kg/m²"
              curr={current.bmi}
              prev={previous.bmi}
              delta={temporal.delta_bmi}
              isFirst={isFirstVisit}
            />
            <DeltaCard
              label="HbA1c"
              unit="%"
              curr={current.hba1c}
              prev={previous.hba1c}
              delta={temporal.delta_hba1c}
              isFirst={isFirstVisit}
            />
            <DeltaCard
              label="Resting Heart Rate"
              unit="bpm"
              curr={current.resting_heart_rate}
              prev={previous.resting_heart_rate}
              delta={temporal.delta_resting_heart_rate}
              isFirst={isFirstVisit}
            />
          </div>
        </div>

        {/* Right: 2 Categorical Changes */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
          {/* Categorical Variables Card */}
          <div className="card" style={{ padding: '1.75rem', borderRadius: '12px', boxShadow: '0 4px 12px rgba(0,0,0,0.05)' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1.25rem', borderBottom: '2px solid #edf2f7', paddingBottom: '0.75rem' }}>
              <Heart size={22} color="#805ad5" />
              <h3 style={{ margin: 0, color: '#1a365d', fontSize: '1.25rem' }}>Categorical Changes (2 Variables)</h3>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
              <CategoricalCard
                label="Smoking Status"
                curr={current.smoking_status || current.smoking}
                prev={previous.smoking_status || previous.smoking}
                changed={temporal.smoking_status_changed}
                isFirst={isFirstVisit}
              />
              <CategoricalCard
                label="Physical Activity"
                curr={current.physical_activity}
                prev={previous.physical_activity}
                changed={temporal.physical_activity_changed}
                isFirst={isFirstVisit}
              />
            </div>
          </div>

        </div>
      </div>

      {/* Chronological Visit Timeline (Section 32) */}
      <div className="card" style={{ marginBottom: '2rem', padding: '2rem', borderRadius: '12px', boxShadow: '0 4px 12px rgba(0,0,0,0.05)' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.25rem', borderBottom: '2px solid #edf2f7', paddingBottom: '0.75rem' }}>
          <div>
            <h3 style={{ margin: 0, color: '#1a365d', fontSize: '1.3rem' }}>Chronological Visit Timeline</h3>
            <p style={{ margin: 0, color: '#718096', fontSize: '0.9rem' }}>PostgreSQL records sorted by assessment date (ORDER BY visit_date ASC)</p>
          </div>
          <span style={{ fontSize: '0.85rem', fontWeight: 600, color: '#2b6cb0', backgroundColor: '#ebf8ff', padding: '0.35rem 0.85rem', borderRadius: '9999px' }}>
            {timeline.length} Visit{timeline.length !== 1 ? 's' : ''} Recorded
          </span>
        </div>

        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '0.92rem' }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #e2e8f0', color: '#4a5568', backgroundColor: '#f7fafc' }}>
                <th style={{ padding: '0.75rem 1rem' }}>Visit #</th>
                <th style={{ padding: '0.75rem 1rem' }}>Date</th>
                <th style={{ padding: '0.75rem 1rem' }}>BP (Sys / Dia)</th>
                <th style={{ padding: '0.75rem 1rem' }}>Cholesterol</th>
                <th style={{ padding: '0.75rem 1rem' }}>LDL / HDL</th>
                <th style={{ padding: '0.75rem 1rem' }}>BMI</th>
                <th style={{ padding: '0.75rem 1rem' }}>HbA1c</th>
                <th style={{ padding: '0.75rem 1rem' }}>Heart Rate</th>
                <th style={{ padding: '0.75rem 1rem' }}>Smoking</th>
                <th style={{ padding: '0.75rem 1rem' }}>Activity</th>
              </tr>
            </thead>
            <tbody>
              {timeline.map((v: any, i: number) => {
                const cv = v.clinical_values || {};
                const isLatest = (i === timeline.length - 1);
                return (
                  <tr key={i} style={{ borderBottom: '1px solid #edf2f7', backgroundColor: isLatest ? '#ebf8ff' : 'transparent' }}>
                    <td style={{ padding: '0.85rem 1rem', fontWeight: 700 }}>
                      Visit {v.visit_number} {isLatest && <span style={{ color: '#2b6cb0', fontSize: '0.8rem' }}>(Latest)</span>}
                    </td>
                    <td style={{ padding: '0.85rem 1rem', color: '#4a5568' }}>{String(v.visit_date).slice(0, 10)}</td>
                    <td style={{ padding: '0.85rem 1rem', fontWeight: 600 }}>
                      {cv.systolic_bp ?? '-'} / {cv.diastolic_bp ?? '-'} mmHg
                    </td>
                    <td style={{ padding: '0.85rem 1rem' }}>{cv.cholesterol ?? '-'} mg/dL</td>
                    <td style={{ padding: '0.85rem 1rem' }}>{cv.ldl ?? '-'} / {cv.hdl ?? '-'}</td>
                    <td style={{ padding: '0.85rem 1rem' }}>{cv.bmi ?? '-'}</td>
                    <td style={{ padding: '0.85rem 1rem' }}>{cv.hba1c ?? '-'}%</td>
                    <td style={{ padding: '0.85rem 1rem' }}>{cv.resting_heart_rate ?? '-'} bpm</td>
                    <td style={{ padding: '0.85rem 1rem' }}>{cv.smoking_status || cv.smoking || '-'}</td>
                    <td style={{ padding: '0.85rem 1rem' }}>{cv.physical_activity ?? '-'}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Action Footer */}
      <div style={{ display: 'flex', justifyContent: 'center', gap: '1rem', marginTop: '2rem' }}>
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
          onClick={() => navigate('/analytics')}
        >
          View Model Analytics & Metrics <ArrowRight size={18} />
        </button>
      </div>
    </div>
  );
}

function DeltaCard({ label, unit, curr, prev, delta, isFirst }: any) {
  const hasPrev = prev !== null && prev !== undefined && String(prev).trim() !== '' && !isFirst;
  const numCurr = curr !== null && curr !== undefined ? Number(curr) : NaN;
  const numPrev = hasPrev ? Number(prev) : NaN;

  let effectiveDelta: number | null = null;
  if (delta !== null && delta !== undefined && !isNaN(Number(delta))) {
    effectiveDelta = Number(delta);
  } else if (hasPrev && !isNaN(numCurr) && !isNaN(numPrev)) {
    effectiveDelta = numCurr - numPrev;
  }

  const hasValidDelta = hasPrev && effectiveDelta !== null && !isNaN(effectiveDelta);
  const deltaVal = effectiveDelta ?? 0;

  return (
    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '0.75rem 1rem', backgroundColor: '#f7fafc', borderRadius: '8px', border: '1px solid #edf2f7' }}>
      <div>
        <div style={{ fontWeight: 700, color: '#2d3748', fontSize: '0.95rem' }}>{label}</div>
        <div style={{ fontSize: '0.82rem', color: '#718096', marginTop: '0.15rem' }}>
          {!hasPrev ? (
            <span>Current: <strong>{curr ?? '-'}</strong> {unit}</span>
          ) : (
            <span>Prev: <strong>{prev}</strong> → Curr: <strong>{curr}</strong> {unit}</span>
          )}
        </div>
      </div>
      <div style={{ textAlign: 'right' }}>
        {!hasValidDelta ? (
          <span style={{ fontSize: '0.85rem', color: '#718096', fontStyle: 'italic', backgroundColor: '#edf2f7', padding: '0.25rem 0.6rem', borderRadius: '6px' }}>
            No change
          </span>
        ) : (
          <span style={{ fontWeight: 800, fontSize: '1.05rem', color: deltaVal > 0 ? '#e53e3e' : deltaVal < 0 ? '#38a169' : '#4a5568' }}>
            Δ {deltaVal > 0 ? `+${deltaVal.toFixed(1)}` : deltaVal.toFixed(1)} {unit}
          </span>
        )}
      </div>
    </div>
  );
}

function CategoricalCard({ label, curr, prev, changed, isFirst }: any) {
  const hasPrev = prev !== null && prev !== undefined && String(prev).trim() !== '' && !isFirst;

  let isChanged: boolean | null = null;
  if (changed !== null && changed !== undefined) {
    isChanged = Boolean(changed);
  } else if (hasPrev && curr !== null && curr !== undefined) {
    isChanged = String(curr).trim().toLowerCase() !== String(prev).trim().toLowerCase();
  }

  return (
    <div style={{ padding: '0.9rem 1rem', backgroundColor: '#f7fafc', borderRadius: '8px', border: '1px solid #edf2f7' }}>
      <div style={{ fontWeight: 700, color: '#2d3748', fontSize: '0.95rem' }}>{label}</div>
      <div style={{ fontSize: '0.82rem', color: '#718096', marginTop: '0.2rem' }}>
        {!hasPrev ? (
          <span>Current: <strong>{curr ?? '-'}</strong></span>
        ) : (
          <span>Prev: <strong>{prev}</strong> → Curr: <strong>{curr}</strong></span>
        )}
      </div>
      <div style={{ marginTop: '0.45rem' }}>
        {!hasPrev || isChanged === null ? (
          <span style={{ fontSize: '0.8rem', color: '#718096', fontStyle: 'italic', backgroundColor: '#edf2f7', padding: '0.2rem 0.5rem', borderRadius: '4px' }}>
            No change
          </span>
        ) : (
          <span style={{ fontSize: '0.82rem', fontWeight: 700, padding: '0.2rem 0.6rem', borderRadius: '4px', backgroundColor: isChanged ? '#feebc8' : '#e2e8f0', color: isChanged ? '#c05621' : '#4a5568' }}>
            Status: {isChanged ? 'Changed' : 'No Change'}
          </span>
        )}
      </div>
    </div>
  );
}
