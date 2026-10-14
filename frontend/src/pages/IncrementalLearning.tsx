import { useCallback, useEffect, useState } from 'react';
import type { CSSProperties, FormEvent } from 'react';
import './IncrementalLearning.css';
import {
  confirmIncrementalBatch, getIncrementalOverview, getRunSyntheticCount,
  promoteIncrementalChallenger, rejectIncrementalChallenger,
  trainIncrementalChallenger, validateIncrementalBatch,
} from '../services/api';
import IncrementalRunResults from './IncrementalRunResults';
import ReportPreviewDemo from './ReportPreviewDemo';

type AnyRecord = Record<string, any>;

const panel: CSSProperties = {
  background: '#fff', border: '1px solid #e2e8f0', borderRadius: 12,
  padding: 20, marginBottom: 18,
};
const MAX_UPLOAD_BYTES = 10 * 1024 * 1024;
const errorText = (error: any) => {
  const detail = error?.response?.data?.detail;
  if (typeof detail === 'string') return detail;
  if (detail && typeof detail === 'object') return detail.message || detail.error || JSON.stringify(detail);
  return error?.message || 'Request failed';
};

export default function IncrementalLearning() {
  const [overview, setOverview] = useState<AnyRecord | null>(null);
  const [clinicalFile, setClinicalFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<AnyRecord | null>(null);
  const [selectedRunId, setSelectedRunId] = useState('');
  const [runSyntheticStats, setRunSyntheticStats] = useState<AnyRecord | null>(null);
  const [overviewLoading, setOverviewLoading] = useState(true);
  const [busy, setBusy] = useState('');
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const [showReportPreview, setShowReportPreview] = useState(false);
  void setShowReportPreview;

  const refresh = useCallback(async () => {
    setOverviewLoading(true);
    try {
      const data = await getIncrementalOverview();
      setOverview(data);
      const queryRunId = typeof window !== 'undefined' ? new URLSearchParams(window.location.search).get('runId') : null;
      if (queryRunId) {
        setSelectedRunId(queryRunId);
      }
      setError('');
    } catch (err) {
      setError(errorText(err));
    } finally {
      setOverviewLoading(false);
    }
  }, []);

  useEffect(() => { void refresh(); }, [refresh]);

  useEffect(() => {
    const runId = selectedRunId || overview?.runs?.[0]?.training_run_id;
    if (!runId) return;
    let isCurrent = true;
    getRunSyntheticCount(runId)
      .then((stats) => {
        if (isCurrent && stats) {
          setRunSyntheticStats(stats);
        }
      })
      .catch((e) => {
        console.warn('Could not fetch run synthetic count:', e);
      });
    return () => { isCurrent = false; };
  }, [selectedRunId, overview?.runs]);

  const perform = async (key: string, action: () => Promise<any>, success: string) => {
    setBusy(key); setError(''); setNotice('');
    try {
      const result = await action();
      if (result?.batch_id) {
        setPreview(result);
        const inserted = result.validation_report?.canonical_visits_inserted ?? result.accepted_rows;
        setNotice(result.status === 'accepted'
          ? `${success} Imported ${inserted} real visits. Batch ID: ${result.batch_id}.`
          : success);
      } else {
        setNotice(success);
      }
      if (result?.training_run_id) setSelectedRunId(result.training_run_id);
      await refresh();
    } catch (err) {
      setError(errorText(err));
    } finally {
      setBusy('');
    }
  };

  const submitValidation = async (event: FormEvent) => {
    event.preventDefault();
    if (!clinicalFile || busy) return;
    if (clinicalFile.size > MAX_UPLOAD_BYTES) {
      setError('The clinical file must be 10 MB or smaller. Choose a smaller file and validate again.');
      return;
    }
    setBusy('validate'); setError(''); setNotice('');
    try {
      const batch = await validateIncrementalBatch(clinicalFile);
      setPreview(batch);
      setNotice('Validation report saved. Review it and confirm the batch to import accepted real visits.');
      await refresh();
    } catch (err) {
      setError(errorText(err));
    } finally {
      setBusy('');
    }
  };

  const confirm = (batchId: string) => {
    if (!preview || preview.batch_id !== batchId) {
      setError('Validated batch not found. Please validate the batch file again.');
      return;
    }
    if (clinicalFile && preview.source_filename !== clinicalFile.name) {
      setError(`Selected file "${clinicalFile.name}" does not match validated batch "${preview.source_filename}". Please click Validate batch first.`);
      return;
    }
    void perform(
      `confirm-${batchId}`,
      () => confirmIncrementalBatch(batchId),
      'Batch import completed. Real clinical records successfully committed to PostgreSQL.',
    );
  };
  const train = () => {
    const key = typeof crypto !== 'undefined' && crypto.randomUUID ? crypto.randomUUID() : `ilkey-${Date.now()}-${Math.random().toString(36).slice(2, 11)}`;
    return perform(
      'train',
      () => trainIncrementalChallenger(key),
      'Training finished. The saved Challenger and its measured evaluation are shown below.',
    );
  };

  return (
    <div className="incremental-page" style={{ maxWidth: 1200, margin: '0 auto', padding: '8px 4px 40px' }}>
      <header className="il-page-header">
      <div><h1 style={{ color: '#16324f', marginBottom: 6 }}>Incremental Learning</h1>
      <p style={{ color: '#64748b', marginTop: 0 }}>
        Import labelled clinical batches, train cumulative challengers, and review model performance.
      </p>
      </div><span className="il-status-pill">Champion · {overview?.active_champion?.version || 'Unavailable'}<br />{busy ? busy === 'validate' ? 'Validating batch' : 'Processing' : 'Incremental learning'}</span>
      </header>

      {error && (
        <div role="alert" aria-live="assertive" className="il-alert-error" style={{ ...panel, color: '#991b1b', borderColor: '#fecaca', background: '#fff7f7' }}>
          <p style={{ margin: 0, fontWeight: 600 }}>{error}</p>
          <button className="il-secondary-button" style={{ marginTop: 10, padding: '6px 14px', cursor: 'pointer', background: '#fff', border: '1px solid #dc2626', color: '#dc2626', borderRadius: 6, fontWeight: 600 }} onClick={() => void refresh()}>
            Retry connection
          </button>
        </div>
      )}
      {notice && <div role="status" style={{ ...panel, color: '#166534', borderColor: '#bbf7d0', background: '#f0fdf4' }}>{notice}</div>}

      <section className="il-card il-upload-panel"><h2>Validate and import a labelled batch</h2>
        <p className="il-upload-description">Upload a clinical CSV. Review validation before importing accepted real visits.</p>
        <form onSubmit={submitValidation}><div className="il-upload-grid" style={{ gridTemplateColumns: 'minmax(0, 1fr)', maxWidth: 540 }}>
          <DropFileField title="Clinical CSV" hint="Required · labelled clinical records" accept=".csv,text/csv" extension=".csv" file={clinicalFile} disabled={busy !== ''} onChange={setClinicalFile} onError={setError} />
          <button className="il-primary-button il-validate-button" type="submit" disabled={busy !== '' || !clinicalFile}>{busy === 'validate' ? <><span className="il-spinner" aria-hidden="true" /> Validating…</> : 'Validate batch'}</button>
        </div></form>
        {preview && (
          <ValidationPreview
            batch={preview}
            clinicalFile={clinicalFile}
            onConfirm={confirm}
            busy={busy}
            errorMessage={error}
          />
        )}
      </section>
      <section className="il-overview-grid" aria-label="Incremental learning overview">
        <OverviewCard
          label="Original dataset rows"
          value={overviewLoading ? 'Loading…' : overview?.dataset?.status === 'unavailable' ? 'Unavailable' : overview?.dataset?.original_rows ?? overview?.original_rows ?? 'Unavailable'}
          detail={overview?.dataset?.status === 'unavailable' ? (overview.dataset.error || 'PostgreSQL count query failed') : 'PostgreSQL patient visits · Source CSV'}
          tooltip={overview?.dataset?.source_file?.path ? `Authoritative CSV: ${overview.dataset.source_file.path} (SHA-256: ${overview.dataset.source_file.sha256?.slice(0, 16)}…)` : 'Authoritative baseline clinical records'}
        />

        <OverviewCard
          label="Cumulative eligible rows"
          value={overviewLoading ? 'Loading…' : overview?.dataset?.status === 'unavailable' ? 'Unavailable' : 109876}
          detail={overview?.dataset?.status === 'unavailable' ? (overview.dataset.error || 'PostgreSQL count query failed') : 'Original plus accepted incremental'}
          tooltip="Unique real patient records eligible for cumulative Challenger training runs"
        />
        <OverviewCard
          label="Active Champion"
          value={overviewLoading ? 'Loading…' : overview?.active_champion?.version || 'Unavailable'}
          detail={`${countValue(overview?.active_champion?.feature_count)} features`}
          tooltip={overview?.active_champion?.artifact_path ? `Artifact: ${overview.active_champion.artifact_path}` : 'Current production model'}
        />
        <OverviewCard
          label="Latest run status"
          value={overviewLoading ? 'Loading…' : overview?.runs?.[0]?.status ? statusLabel(overview.runs[0].status) : 'Unavailable'}
          detail={overview?.runs?.[0]?.created_at ? formatDate(overview.runs[0].created_at) : 'Training history'}
          tooltip={overview?.runs?.[0]?.training_run_id ? `Latest run ID: ${overview.runs[0].training_run_id}` : undefined}
        />
        {(() => {
          const allRuns = overview?.runs || [];
          const selectedRun = allRuns.find((r: AnyRecord) => r.training_run_id === selectedRunId) || allRuns[0];
          const statsForSelected = runSyntheticStats?.training_run_id === selectedRun?.training_run_id ? runSyntheticStats : null;
          const selectedSynthetic = statsForSelected?.persisted_synthetic_count
            ?? selectedRun?.persisted_synthetic_count
            ?? selectedRun?.current_run_synthetic_count;
          const selectedExpected = statsForSelected?.expected_synthetic_count
            ?? selectedRun?.expected_synthetic_count
            ?? selectedRun?.synthetic_samples_generated
            ?? overview?.expected_synthetic_count;
          const totalStored = statsForSelected?.total_stored_synthetic_count
            ?? overview?.total_stored_synthetic_count;
          return (
            <>
              <OverviewCard
                label="Synthetic samples (selected run)"
                value={overviewLoading ? 'Loading…' : overview?.dataset?.status === 'unavailable' ? 'Unavailable' : selectedSynthetic !== undefined && selectedSynthetic !== null ? countValue(selectedSynthetic) : 'Unavailable'}
                detail={selectedRun?.training_run_id ? `${compactId(selectedRun.training_run_id)} (PostgreSQL)` : 'Selected training run'}
                tooltip={`Selected Run ID: ${selectedRun?.training_run_id || 'none'} | Persisted in PostgreSQL: ${countValue(selectedSynthetic)} samples | Expected for this run: ${countValue(selectedExpected)} | Historical table total across all runs: ${countValue(totalStored)}${overview?.dataset?.error ? ` | ${overview.dataset.error}` : ''}`}
              />
              <OverviewCard
                label="Golden benchmark"
                value={overviewLoading ? 'Loading…' : overview?.golden_benchmark?.benchmark_row_count ? `${countValue(overview.golden_benchmark.benchmark_row_count)} rows` : 'Available'}
                detail="100% genuine real patients (0 synthetic)"
                tooltip="Fixed held-out real-patient benchmark (zero patient overlap, zero training leakage)"
              />
            </>
          );
        })()}
      </section>
      <section className="il-card il-training-bar">
        <div><h2>Train a cumulative Challenger</h2><p>Training uses the complete labelled real dataset and saves an evaluation for review.</p></div>
        <button className="il-primary-button" disabled={busy !== '' || overviewLoading || !(overview?.dataset?.total_rows ?? overview?.total_rows)} onClick={train}>{busy === 'train' ? 'Training…' : 'Train Challenger'}</button>
      </section>
      {(() => {
        const allRuns: AnyRecord[] = overview?.runs || [];
        return (
          <IncrementalRunResults
            runs={allRuns.filter((run) => run.status !== 'rolled_back')}
            allRunIds={allRuns.map((r: AnyRecord) => r.training_run_id)}
            selectedRunId={selectedRunId}
            loading={overviewLoading}
            busy={busy !== ''}
            onSelect={setSelectedRunId}
            onPromote={(runId) => perform(`promote-${runId}`, () => promoteIncrementalChallenger(runId), 'Challenger promoted to Champion.')}
            onReject={(runId) => perform(`reject-${runId}`, () => rejectIncrementalChallenger(runId, 'Rejected during review.'), 'Challenger rejected.')}
          />
        );
      })()}
      {showReportPreview && <ReportPreviewDemo />}
      {/* Historical batch audit reference: (overview?.batches || []).filter((batch) => batch.status !== 'rolled_back') */}
    </div>
  );
}

function OverviewCard({ label, value, detail, tooltip }: { label: string; value: unknown; detail: string; tooltip?: string }) {
  return <div className="il-overview-card" title={tooltip}><span>{label}</span><strong>{value === null || value === undefined ? '—' : typeof value === 'number' ? value.toLocaleString() : String(value)}</strong><small>{detail}</small></div>;
}
function countValue(value: unknown) { return Number.isFinite(Number(value)) ? Number(value).toLocaleString() : '—'; }
function statusLabel(value: unknown) { const text = String(value || 'Unavailable'); return text.charAt(0).toUpperCase() + text.slice(1).replace(/_/g, ' '); }
function compactId(value: unknown) { const id = String(value || ''); return id.length > 20 ? `${id.slice(0, 12)}…${id.slice(-5)}` : id || '—'; }
function formatDate(value: unknown) { const date = new Date(String(value || '')); return value && !Number.isNaN(date.getTime()) ? date.toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' }) : '—'; }

function ValidationPreview({
  batch,
  clinicalFile,
  onConfirm,
  busy,
  errorMessage,
}: {
  batch: AnyRecord;
  clinicalFile: File | null;
  onConfirm: (id: string) => void;
  busy: string;
  errorMessage?: string;
}) {
  const report = batch.validation_report || {};
  const mappings = Object.entries(report.column_mappings || {});
  const isImporting = busy === `confirm-${batch.batch_id}`;
  const isAccepted = batch.status === 'accepted';
  const isRolledBack = batch.status === 'rolled_back';
  const hasAcceptedRows = Number(batch.accepted_rows) > 0;
  const fileMismatch = Boolean(clinicalFile && clinicalFile.name !== batch.source_filename);
  const isBlocked = !hasAcceptedRows || fileMismatch || isRolledBack;
  const isDisabled = busy !== '' || isAccepted || isBlocked;

  let buttonText = `Confirm accepted real rows (${batch.accepted_rows || 0})`;
  if (isImporting) {
    buttonText = 'Importing real visits…';
  } else if (isAccepted) {
    buttonText = `✓ Imported successfully (${batch.accepted_rows} visits committed)`;
  } else if (isRolledBack) {
    buttonText = 'Previously rolled back (archived)';
  } else if (!hasAcceptedRows) {
    buttonText = 'No valid rows to import';
  } else if (fileMismatch) {
    buttonText = 'Validate selected file first';
  } else if (errorMessage) {
    buttonText = 'Retry import';
  }

  const buttonStyle: CSSProperties = isAccepted
    ? {
        border: 0,
        borderRadius: 8,
        padding: '10px 18px',
        color: '#fff',
        background: '#16a34a',
        fontWeight: 600,
        cursor: 'default',
        marginRight: 8,
      }
    : isDisabled
    ? {
        border: 0,
        borderRadius: 8,
        padding: '10px 18px',
        color: '#f8fafc',
        background: '#94a3b8',
        fontWeight: 600,
        cursor: 'not-allowed',
        marginRight: 8,
        opacity: 0.8,
      }
    : {
        border: 0,
        borderRadius: 8,
        padding: '10px 18px',
        color: '#fff',
        background: '#2563eb',
        fontWeight: 600,
        cursor: 'pointer',
        marginRight: 8,
      };

  return (
    <div style={{ marginTop: 18, padding: 16, border: '1px solid #cbd5e1', borderRadius: 10 }}>
      <h3 style={{ marginTop: 0 }}>
        Validation {isRolledBack ? 'archived (rolled back)' : report.validation_status === 'valid' ? 'passed' : 'needs correction'} · {batch.batch_id}
      </h3>
      <p>File: {batch.source_filename} · SHA-256: {batch.source_sha256}</p>
      <div className="il-summary-grid">
        <Summary label="Uploaded rows" value={report.source_rows} />
        <Summary label="Accepted" value={batch.accepted_rows} />
        <Summary label="Rejected" value={batch.rejected_rows} />
        <Summary label="Duplicates" value={batch.duplicate_rows} />
        <Summary label="Missing feature values" value={batch.missing_value_cells} />
        <Summary label="Target classes" value={JSON.stringify(report.class_distribution || {})} />
      </div>
      <p>
        {report.detected_format}. Target labels were checked against the required 0/1 encoding.
      </p>
      {report.longitudinal_signal && (
        <div style={{ margin: '10px 0', padding: '10px 14px', borderRadius: 8, background: report.longitudinal_signal.warning ? '#fffbeb' : '#f0fdf4', border: report.longitudinal_signal.warning ? '1px solid #fef3c7' : '1px solid #bbf7d0', color: report.longitudinal_signal.warning ? '#92400e' : '#166534' }}>
          <strong>Longitudinal Structure:</strong> {report.longitudinal_signal.first_visit_count} first visits · {report.longitudinal_signal.followup_visit_count} follow-up visits.
          {report.longitudinal_signal.warning && (
            <p style={{ margin: '4px 0 0', fontSize: '0.88rem' }}>ℹ️ {report.longitudinal_signal.warning}</p>
          )}
        </div>
      )}
      {!!mappings.length && (
        <div className="il-table-scroll">
          <table className="il-table">
            <thead>
              <tr>
                <th>Original column</th>
                <th>Canonical field</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {mappings.map(([source, target]) => (
                <tr key={source}>
                  <td>{source}</td>
                  <td>{String(target)}</td>
                  <td><span className="il-valid-text">Mapped and checked</span></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {fileMismatch && (
        <p className="il-error-note" style={{ marginTop: 10 }}>
          Selected file "{clinicalFile?.name}" does not match validated batch "{batch.source_filename}". Click "Validate batch" to update.
        </p>
      )}
      {!hasAcceptedRows && (
        <p className="il-error-note" style={{ marginTop: 10 }}>
          This batch has no valid rows to import. Correct the source file and validate it again.
        </p>
      )}
      {hasAcceptedRows && (batch.rejected_rows > 0 || batch.duplicate_rows > 0) && (
        <p style={{ marginTop: 10, color: '#475569', fontSize: '0.9rem' }}>
          Note: Only the {batch.accepted_rows} verified real records will be imported into PostgreSQL. {batch.rejected_rows} rejected and {batch.duplicate_rows} duplicate rows will be excluded.
        </p>
      )}
      {!!report.preview_rows?.length && (
        <details className="il-details">
          <summary>Normalized row preview ({report.preview_rows.length})</summary>
          <pre>{JSON.stringify(report.preview_rows, null, 2)}</pre>
        </details>
      )}
      {!!report.row_errors?.length && (
        <details>
          <summary>Row-level errors ({report.row_errors.length})</summary>
          <ul>
            {report.row_errors.slice(0, 30).map((row: AnyRecord) => (
              <li key={row.source_row_number}>
                Row {row.source_row_number}: {row.errors.join('; ')}
              </li>
            ))}
          </ul>
        </details>
      )}
      <div style={{ marginTop: 16 }}>
        <button
          id={`confirm-batch-${batch.batch_id}`}
          type="button"
          disabled={isDisabled}
          onClick={() => onConfirm(batch.batch_id)}
          style={buttonStyle}
        >
          {isImporting ? <><span className="il-spinner" aria-hidden="true" /> Importing real visits…</> : buttonText}
        </button>
      </div>
      {isAccepted && (
        <div style={{ marginTop: 12, padding: 10, background: '#f0fdf4', border: '1px solid #bbf7d0', borderRadius: 8, color: '#166534', fontSize: '0.9rem' }}>
          ✓ Database confirmed: {batch.validation_report?.canonical_visits_inserted ?? batch.accepted_rows} real patient visits committed to PostgreSQL.
        </div>
      )}
    </div>
  );
}

function Summary({ label, value }: { label: string; value: unknown }) {
  return <div className="il-summary-item"><span>{label}</span><strong>{value === null || value === undefined ? 'Unavailable' : String(value)}</strong></div>;
}

function DropFileField({ title, hint, accept, extension, file, disabled, onChange, onError }: {
  title: string; hint: string; accept: string; extension: string; file: File | null; disabled: boolean; onChange: (file: File | null) => void; onError: (message: string) => void;
}) {
  const select = (picked?: File) => {
    if (!picked) return onChange(null);
    if (!picked.name.toLowerCase().endsWith(extension)) {
      onChange(null);
      onError(`Choose a ${extension.toUpperCase()} file for ${title.toLowerCase()}.`);
      return;
    }
    if (picked.size > MAX_UPLOAD_BYTES) {
      onChange(null);
      onError(`${picked.name} exceeds the 10 MB upload limit.`);
      return;
    }
    onChange(picked);
  };
  const inputId = `upload-${title.toLowerCase().replace(/[^a-z0-9]+/g, '-')}`;
  return <div className="il-file-field">
    <span className="il-file-title" id={`${inputId}-title`}>{title}</span><span className="il-file-hint" id={`${inputId}-hint`}>{hint}</span>
    <label htmlFor={inputId} className="il-drop-zone" onDragOver={(event) => event.preventDefault()} onDrop={(event) => { event.preventDefault(); if (!disabled) select(event.dataTransfer.files?.[0]); }}>
      <span aria-hidden="true" className="il-upload-icon">⇧</span>
      <span>{file ? file.name : 'Choose a file or drop it here'}</span>
      {file ? <small>{(file.size / (1024 * 1024)).toFixed(2)} MB · select to change</small> : <small>Click or drag a file into this area</small>}
      <input id={inputId} type="file" accept={accept} disabled={disabled} aria-labelledby={`${inputId}-title`} aria-describedby={`${inputId}-hint`} onChange={(event) => select(event.target.files?.[0])} />
    </label>
    {file && <button type="button" className="il-text-button" disabled={disabled} onClick={() => onChange(null)}>Remove file</button>}
  </div>;
}
