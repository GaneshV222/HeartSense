import { useMemo, useState } from 'react';
import {
  CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts';

type RecordValue = Record<string, any>;
type Props = {
  runs: RecordValue[];
  selectedRunId: string;
  allRunIds?: string[];
  loading: boolean;
  busy: boolean;
  onSelect: (runId: string) => void;
  onPromote: (runId: string) => void;
  onReject: (runId: string) => void;
};

const PAGE_SIZE = 8;
const metricRows: [string, string][] = [
  ['accuracy', 'Accuracy'], ['precision', 'Precision'], ['recall_sensitivity', 'Sensitivity / Recall'],
  ['specificity', 'Specificity'], ['f1', 'F1-score'], ['roc_auc', 'ROC-AUC'], ['pr_auc', 'PR-AUC'],
];
const count = (value: unknown) => Number.isFinite(Number(value)) ? Number(value).toLocaleString() : '—';
const recordedCount = (value: unknown) => Number.isFinite(Number(value)) ? Number(value).toLocaleString() : 'Not recorded';
const asPercent = (value: unknown) => Number.isFinite(Number(value)) ? `${(Number(value) * 100).toFixed(2)}%` : '—';
const asPoints = (value: unknown) => {
  if (!Number.isFinite(Number(value))) return 'Not comparable';
  const points = Number(value) * 100;
  const sign = points > 0 ? '+' : '';
  const dir = points > 0 ? 'higher' : points < 0 ? 'lower' : 'unchanged';
  return `${sign}${points.toFixed(2)} pp (${dir})`;
};
const statusText = (value: unknown) => {
  const key = String(value || 'unavailable').toLowerCase();
  return ({ completed: 'Completed', failed: 'Failed', running: 'Running', validated: 'Validated', accepted: 'Imported', rolled_back: 'Rolled Back' } as Record<string, string>)[key]
    || key.charAt(0).toUpperCase() + key.slice(1);
};
const dateText = (value: unknown) => {
  if (!value) return '—';
  const date = new Date(String(value));
  return Number.isNaN(date.getTime()) ? '—' : date.toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' });
};
const shortId = (value: string) => value.length > 16 ? `${value.slice(0, 11)}…${value.slice(-4)}` : value;
const totals = (values: RecordValue) => Object.values(values || {}).reduce((sum: number, item: any) => sum + (Number(item) || 0), 0);

export default function IncrementalRunResults({ runs, selectedRunId, allRunIds, loading, busy, onSelect, onPromote, onReject }: Props) {
  const [page, setPage] = useState(0);
  const [matrixModel, setMatrixModel] = useState<'challenger' | 'champion'>('challenger');
  const [protocolView, setProtocolView] = useState<'golden' | 'diagnostic'>('golden');
  const selected = runs.find((run) => run.training_run_id === selectedRunId);
  const start = page * PAGE_SIZE;
  const visibleRuns = runs.slice(start, start + PAGE_SIZE);
  const pages = Math.max(1, Math.ceil(runs.length / PAGE_SIZE));
  const report = selected?.report || {};
  const metrics = report.optimized?.test_metrics || {};

  if (selectedRunId && !selected) {
    const isHidden = allRunIds?.includes(selectedRunId);
    return (
      <section className="il-card il-run-detail">
        <p className="il-muted">
          Run <code>{selectedRunId}</code> {isHidden ? 'persisted but hidden by the rolled-back history filter' : 'is not present in PostgreSQL run history'}.
        </p>
      </section>
    );
  }

  const hasGolden = Boolean(report.golden_benchmark_comparison?.comparable);
  const activeProtocol = protocolView === 'golden' && hasGolden ? 'golden' : 'diagnostic';
  const comparison = activeProtocol === 'golden'
    ? (report.golden_benchmark_comparison || report.champion_comparison || {})
    : (report.champion_comparison || {});

  const smote = report.smote || {};
  const featureInfo = report.feature_schema || {};
  const comparable = comparison.comparable === true;
  const chosenMatrix = matrixModel === 'champion' && comparable
    ? comparison.champion_metrics?.confusion_matrix?.matrix
    : (comparable ? comparison.challenger_metrics?.confusion_matrix?.matrix : metrics.confusion_matrix?.matrix);
  const beforeTotal = totals(smote.before);
  const afterTotal = totals(smote.after);
  const classes = [...new Set([...Object.keys(smote.before || {}), ...Object.keys(smote.after || {})])].sort();

  const curveData = useMemo(() => {
    const curves = activeProtocol === 'golden'
      ? (selected?.golden_curves || selected?.comparison_curves)
      : selected?.comparison_curves;
    if (!comparison.comparable || !curves) return { roc: [], pr: [] };
    const roc = curves.roc?.challenger_false_positive_rate?.map((x: number, i: number) => ({
      x, challenger: curves.roc.challenger_true_positive_rate[i], champion: curves.roc.champion_true_positive_rate[i],
    })) || [];
    const pr = curves.precision_recall?.challenger_recall?.map((x: number, i: number) => ({
      x, challenger: curves.precision_recall.challenger_precision[i], champion: curves.precision_recall.champion_precision[i],
    })) || [];
    return { roc, pr };
  }, [selected, comparison.comparable, activeProtocol]);

  return <>
    <section className="il-card il-run-history">
      <div className="il-section-heading"><div><h2>Training run history</h2><p>Choose a run to view its persisted evaluation and decision.</p></div><span>{runs.length.toLocaleString()} runs</span></div>
      {loading && runs.length === 0 ? <p className="il-muted" role="status">Loading training history…</p> : runs.length === 0 ? <p className="il-muted">No training runs are available yet.</p> : <>
        <div className="il-table-scroll"><table className="il-table il-run-table"><thead><tr>
          <th>Run ID</th><th>Date</th><th className="il-num">Real visits</th><th className="il-num">New records</th><th className="il-num">Synthetic in run</th><th className="il-num">Accuracy</th><th className="il-num">Recall</th><th className="il-num">F1</th><th className="il-num">ROC-AUC</th><th>Status</th><th>Decision</th>
        </tr></thead><tbody>
          {visibleRuns.map((run) => {
            const test = run.report?.optimized?.test_metrics || {};
            const isSelected = run.training_run_id === selected?.training_run_id;
            const synthCount = run.synthetic_samples_generated;
            const newRows = Number(run.new_records_added) || 0;
            return <tr key={run.training_run_id} className={isSelected ? 'il-selected-row' : ''} onClick={() => onSelect(run.training_run_id)} tabIndex={0} onKeyDown={(event) => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); onSelect(run.training_run_id); } }} aria-selected={isSelected}>
              <td title={run.training_run_id}><button className="il-run-link" onClick={(event) => { event.stopPropagation(); onSelect(run.training_run_id); }}>{shortId(run.training_run_id)}</button></td>
              <td>{dateText(run.created_at)}</td>
              <td className="il-num">{count(run.real_visits ?? run.dataset_rows)}</td>
              <td className="il-num" title="New unique real patient visits introduced in this training run">
                {newRows > 0 ? (
                  <span style={{ color: '#2563eb', fontWeight: 600 }}>+{count(newRows)}</span>
                ) : (
                  <span style={{ color: '#64748b' }}>0 (Baseline)</span>
                )}
              </td>
              <td className="il-num" title={`Synthetic samples generated for run ${run.training_run_id} (used for training)`}>{count(synthCount)}</td>
              <td className="il-num">{asPercent(test.accuracy)}</td>
              <td className="il-num">{asPercent(test.recall_sensitivity ?? test.recall ?? test.sensitivity)}</td>
              <td className="il-num">{asPercent(test.f1)}</td>
              <td className="il-num">{asPercent(test.roc_auc)}</td>
              <td><StatusBadge value={run.status} /></td>
              <td title={run.decision_reason || ''}><StatusBadge value={run.decision || 'pending'} /></td>
            </tr>;
          })}
        </tbody></table></div>
        {pages > 1 && <div className="il-pagination"><span>Showing {start + 1}–{Math.min(start + PAGE_SIZE, runs.length)} of {runs.length}</span><div><button disabled={page === 0} onClick={() => setPage(page - 1)}>Previous</button><span>Page {page + 1} of {pages}</span><button disabled={page + 1 >= pages} onClick={() => setPage(page + 1)}>Next</button></div></div>}
      </>}
    </section>

    {selected && <section className="il-card il-run-detail" aria-live="polite">
      <div className="il-section-heading">
        <div>
          <h2>Selected run <span className="il-mono">{shortId(selected.training_run_id)}</span></h2>
          <p title={selected.training_run_id}>{statusText(selected.status)} · {dateText(selected.created_at)}</p>
        </div>
        <StatusBadge value={selected.decision || 'pending'} />
      </div>

      <div className="il-summary-grid" style={{ marginBottom: 14 }}>
        <div className="il-summary-item">
          <span>Real visits trained</span>
          <strong>{count(selected.real_visits ?? selected.dataset_rows)}</strong>
        </div>
        <div className="il-summary-item">
          <span>Unique real patients</span>
          <strong>{count(selected.unique_real_patients ?? selected.dataset_rows)}</strong>
        </div>
        <div className="il-summary-item">
          <span>New records added</span>
          <strong style={{ color: Number(selected.new_records_added) > 0 ? '#2563eb' : '#1e293b' }}>
            {Number(selected.new_records_added) > 0 ? `+${count(selected.new_records_added)}` : '0 (Baseline)'}
          </strong>
        </div>
        <div className="il-summary-item">
          <span>Synthetic generated (run)</span>
          <strong>{recordedCount(selected.synthetic_samples_generated)}</strong>
        </div>
        <div className="il-summary-item">
          <span>Synthetic used in training</span>
          <strong>{selected.synthetic_samples_used_for_training != null ? count(selected.synthetic_samples_used_for_training) : 'Not recorded'}</strong>
        </div>
        <div className="il-summary-item">
          <span>Evaluation benchmark</span>
          <strong>{activeProtocol === 'golden' ? 'Golden Benchmark v1.0 (Authoritative)' : 'Diagnostic 20% Split (Historical)'}</strong>
        </div>
      </div>

      {selected.status === 'rolled_back' && (
        <div style={{
          padding: '10px 14px',
          borderRadius: 8,
          marginBottom: 14,
          background: '#fef3c7',
          border: '1px solid #fde68a',
          color: '#92400e',
          fontSize: '0.86rem'
        }}>
          ⚠️ <strong>Historical Run (Rolled Back):</strong> This persisted run is marked rolled back. Its run artifacts and historical decision remain separate from the current active dataset; see the recorded lineage for the exact rollback scope.
        </div>
      )}

      {selected.decision_reason && (
        <div style={{
          padding: '10px 14px',
          borderRadius: 8,
          marginBottom: 14,
          background: selected.decision === 'rejected' ? '#fff7f7' : selected.decision === 'promoted' ? '#f0fdf4' : '#f8fafc',
          border: selected.decision === 'rejected' ? '1px solid #fecaca' : selected.decision === 'promoted' ? '1px solid #bbf7d0' : '1px solid #e2e8f0',
          color: selected.decision === 'rejected' ? '#991b1b' : selected.decision === 'promoted' ? '#166534' : '#334155'
        }}>
          <strong>{selected.decision === 'rejected' ? 'Rejection Reason:' : selected.decision === 'promoted' ? 'Promotion Policy:' : 'Promotion Review Status:'}</strong>{' '}
          {selected.decision_reason}
        </div>
      )}

      {(selected.status === 'completed' || selected.status === 'rolled_back') ? <>
        <div style={{ display: 'flex', gap: 10, margin: '10px 0 14px', alignItems: 'center', flexWrap: 'wrap' }}>
          <span style={{ fontWeight: 600, fontSize: '0.88rem', color: '#475569' }}>Evaluation Protocol:</span>
          <button
            style={{
              padding: '6px 14px', borderRadius: 8, border: '1px solid #cbd5e1', cursor: 'pointer',
              background: activeProtocol === 'golden' ? '#2563eb' : '#fff',
              color: activeProtocol === 'golden' ? '#fff' : '#1e293b',
              fontWeight: 600, fontSize: '0.84rem'
            }}
            onClick={() => setProtocolView('golden')}
          >
            🛡️ Authoritative Protocol: Golden Benchmark (15,006 Real Patients · Leakage-Free)
          </button>
          <button
            style={{
              padding: '6px 14px', borderRadius: 8, border: '1px solid #cbd5e1', cursor: 'pointer',
              background: activeProtocol === 'diagnostic' ? '#2563eb' : '#fff',
              color: activeProtocol === 'diagnostic' ? '#fff' : '#1e293b',
              fontWeight: 600, fontSize: '0.84rem'
            }}
            onClick={() => setProtocolView('diagnostic')}
          >
            🔬 Diagnostic Only: SMOTE-before-split (Non-Authoritative)
          </button>
        </div>
        {activeProtocol === 'diagnostic' && (
          <div style={{ padding: '8px 12px', background: '#fffbeb', border: '1px solid #fef3c7', borderRadius: 6, margin: '6px 0 10px', fontSize: '0.82rem', color: '#b45309' }}>
            ⚠️ <strong>Diagnostic Protocol Notice:</strong> SMOTE-before-split metrics are preserved strictly for historical diagnostic reference and cannot override or determine model promotion decisions. Only the Golden Benchmark is authoritative.
          </div>
        )}
        {comparison.historical_champion_independence_audit?.leakage_certification && (
          <div style={{ padding: '8px 12px', background: '#f8fafc', border: '1px solid #e2e8f0', borderRadius: 6, margin: '8px 0 12px', fontSize: '0.82rem', color: '#475569' }}>
            ℹ️ <strong>Provenance Audit:</strong> {comparison.historical_champion_independence_audit.leakage_certification}
          </div>
        )}
        <p className="il-protocol-caption">{comparison.evaluation_protocol || report.evaluation_protocol || 'Evaluation protocol unavailable.'}</p>
        <div className="il-table-scroll"><table className="il-table il-comparison-table"><thead><tr>
          <th>Metric</th>
          <th className="il-num">Champion {comparison.champion_model ? `(${comparison.champion_model})` : ''}</th>
          <th className="il-num">Challenger {report.optimized?.model ? `(${report.optimized.model})` : ''}</th>
          <th className="il-num">Change</th>
        </tr></thead><tbody>
          {metricRows.map(([key, label]) => <tr key={key}><th scope="row">{label}</th>
            <td className="il-num">{comparable ? asPercent(comparison.champion_metrics?.[key]) : 'Not comparable'}</td>
            <td className="il-num">{asPercent(comparable ? comparison.challenger_metrics?.[key] : metrics[key])}</td>
            <td className="il-num">{comparable ? asPoints(comparison.metric_deltas?.[key]) : 'Not comparable'}</td>
          </tr>)}
        </tbody></table></div>
        {!comparable && <p className="il-muted">Champion comparison unavailable: {comparison.reason || 'the persisted run has no compatible same-row comparison.'}</p>}
        {comparable && <p className="il-muted">Change is shown in percentage points. Review sensitivity and specificity alongside accuracy under the configured acceptance policy.</p>}

        <div className="il-results-grid">
          <section className="il-subpanel"><div className="il-section-heading"><h3>Confusion matrix</h3>{comparable && <div className="il-toggle" role="group" aria-label="Confusion matrix model">
            <button aria-pressed={matrixModel === 'challenger'} onClick={() => setMatrixModel('challenger')}>Challenger</button><button aria-pressed={matrixModel === 'champion'} onClick={() => setMatrixModel('champion')}>Champion</button>
          </div>}</div>
            <ConfusionMatrix matrix={chosenMatrix} />
            <small className="il-muted">TN: true negative · FP: false positive · FN: false negative · TP: true positive</small>
          </section>
          <section className="il-subpanel"><h3>SMOTENC class counts</h3>
            {classes.length ? <div className="il-table-scroll"><table className="il-table il-count-table"><thead><tr><th>Class</th><th className="il-num">Before</th><th className="il-num">After</th><th className="il-num">Synthetic added</th></tr></thead><tbody>
              {classes.map((key) => <tr key={key}><th scope="row">Class {key}</th><td className="il-num">{count(smote.before?.[key])}</td><td className="il-num">{count(smote.after?.[key])}</td><td className="il-num">{count((Number(smote.after?.[key]) || 0) - (Number(smote.before?.[key]) || 0))}</td></tr>)}
              <tr className="il-total-row"><th>Total</th><td className="il-num">{count(beforeTotal)}</td><td className="il-num">{count(afterTotal)}</td><td className="il-num">{count(afterTotal - beforeTotal)}</td></tr>
            </tbody></table></div> : <p className="il-muted">Class counts are unavailable for this run.</p>}
            <p className="il-count-caption">{count(smote.samples_before)} real rows · {count(smote.samples_after)} resampled · {count(smote.synthetic_samples_created)} synthetic generated</p>
            {(() => {
              const storage = report.smote_database_storage || report.smote?.database_storage;
              const generated = Number(smote.synthetic_samples_created) || 0;
              const persisted = selected?.persisted_synthetic_count !== undefined
                ? Number(selected.persisted_synthetic_count)
                : storage?.verified_persisted_rows !== undefined
                ? Number(storage.verified_persisted_rows)
                : smote.synthetic_rows_persisted_to_database !== undefined
                ? Number(smote.synthetic_rows_persisted_to_database)
                : undefined;
              if (persisted !== undefined) {
                if (persisted !== generated && generated > 0) {
                  return (
                    <p style={{ color: '#dc2626', fontWeight: 600, fontSize: '0.85rem', marginTop: 4 }}>
                      ⚠️ Storage Discrepancy: {count(generated)} generated vs {count(persisted)} persisted in {storage?.table || 'synthetic_training_samples'}.
                    </p>
                  );
                }
                const newlyInserted = selected?.synthetic_samples_newly_inserted;
                const historicalCount = selected?.synthetic_samples_already_persisted;
                const detailText = (newlyInserted !== undefined && historicalCount !== undefined && Number(historicalCount) > 0)
                  ? ` (${count(newlyInserted)} newly inserted for this run + ${count(historicalCount)} historical baseline preserved)`
                  : '';
                return (
                  <p style={{ color: '#16a34a', fontWeight: 600, fontSize: '0.85rem', marginTop: 4 }}>
                    ✓ Reconciled: All {count(persisted)} synthetic samples for this run persisted in PostgreSQL ({storage?.table || 'synthetic_training_samples'}){detailText}. Historical runs preserved.
                  </p>
                );
              }
              return (
                <p style={{ color: '#64748b', fontSize: '0.85rem', marginTop: 4 }}>
                  PostgreSQL persistence: Not stored in database table for this run (artifact only).
                </p>
              );
            })()}
          </section>
        </div>

        {comparable && <details className="il-technical-details il-curves"><summary>ROC and precision-recall curves</summary>
          <div className="il-curve-legend"><span><i className="il-legend-champion" /> Champion</span><span><i className="il-legend-challenger" /> Challenger</span></div>
          <div className="il-curves-grid">
            <div><h4>ROC</h4><div className="il-chart"><ResponsiveContainer width="100%" height="100%"><LineChart data={curveData.roc}><CartesianGrid strokeDasharray="3 3" /><XAxis dataKey="x" type="number" domain={[0, 1]} /><YAxis type="number" domain={[0, 1]} /><Tooltip /><Line name="Champion" dataKey="champion" dot={false} stroke="#64748b" isAnimationActive={false} /><Line name="Challenger" dataKey="challenger" dot={false} stroke="#2563eb" isAnimationActive={false} /></LineChart></ResponsiveContainer></div></div>
            <div><h4>Precision-recall</h4><div className="il-chart"><ResponsiveContainer width="100%" height="100%"><LineChart data={curveData.pr}><CartesianGrid strokeDasharray="3 3" /><XAxis dataKey="x" type="number" domain={[0, 1]} /><YAxis type="number" domain={[0, 1]} /><Tooltip /><Line name="Champion" dataKey="champion" dot={false} stroke="#64748b" isAnimationActive={false} /><Line name="Challenger" dataKey="challenger" dot={false} stroke="#2563eb" isAnimationActive={false} /></LineChart></ResponsiveContainer></div></div>
          </div>
        </details>}

      </> : <p className="il-muted" style={{ margin: '14px 0' }}>This run has no completed persisted evaluation.</p>}

      <div className="il-run-actions">
        <p><strong>Decision:</strong> {report.promotion_eligibility?.reason || selected.decision_reason || (selected.status === 'failed' ? 'Training failed.' : 'Pending review.')}</p>
        <div>
          <button
            className="il-primary-button"
            disabled={busy || selected.decision === 'promoted' || !report.promotion_eligibility?.eligible}
            onClick={() => onPromote(selected.training_run_id)}
            title={selected.decision === 'promoted' ? 'Already promoted to Champion' : !report.promotion_eligibility?.eligible ? 'Challenger did not pass Golden Benchmark policy' : 'Promote this Challenger to active Champion'}
          >
            Promote Challenger
          </button>
          <button
            className="il-secondary-button"
            disabled={busy || selected.decision === 'rejected' || selected.decision === 'promoted'}
            onClick={() => onReject(selected.training_run_id)}
            title={selected.decision === 'promoted' ? 'Promoted Champion cannot be rejected' : selected.decision === 'rejected' ? 'Already rejected' : 'Reject this Challenger'}
          >
            Reject Challenger
          </button>
        </div>
      </div>

      <RunDetails run={selected} report={report} smote={smote} featureInfo={featureInfo} comparison={comparison} />
    </section>}
  </>;
}

function StatusBadge({ value }: { value: unknown }) {
  const label = statusText(value);
  return <span className={`il-status-badge il-status-${String(value || '').toLowerCase()}`}>{label}</span>;
}

function ConfusionMatrix({ matrix }: { matrix: number[][] | undefined }) {
  if (!Array.isArray(matrix) || matrix.length < 2 || matrix.some((row) => !Array.isArray(row) || row.length < 2)) {
    return <p className="il-muted">Confusion matrix unavailable.</p>;
  }
  const total = (matrix[0][0] || 0) + (matrix[0][1] || 0) + (matrix[1][0] || 0) + (matrix[1][1] || 0);
  return <>
    <table className="il-table il-matrix-table"><thead><tr><th>Actual / Predicted</th><th className="il-num">Predicted 0</th><th className="il-num">Predicted 1</th></tr></thead><tbody>
      <tr><th scope="row">Actual 0</th><td className="il-num">{count(matrix[0][0])}</td><td className="il-num">{count(matrix[0][1])}</td></tr>
      <tr><th scope="row">Actual 1</th><td className="il-num">{count(matrix[1][0])}</td><td className="il-num">{count(matrix[1][1])}</td></tr>
    </tbody></table>
    <small className="il-muted" style={{ display: 'block', marginTop: 4 }}>Total evaluated in matrix: {count(total)} records</small>
  </>;
}

function RunDetails({ run, report, smote, featureInfo, comparison }: RecordValue) {
  const schema = featureInfo.features || [];
  const settings = smote.sampler_configuration || featureInfo.sampler_configuration || {};
  const cats = featureInfo.categorical_features || [];
  const evaluationRows = comparison.evaluation_rows || report.evaluation_artifacts?.prediction_rows;
  const testSynthetic = comparison.synthetic_rows_in_test ?? report.evaluation_artifacts?.synthetic_rows_in_test;
  const testReal = (Number(evaluationRows) || 0) - (Number(testSynthetic) || 0);
  return <details className="il-technical-details"><summary>Run details and sampler configuration</summary>
    <div className="il-detail-grid">
      <Detail label="Full run ID" value={run.training_run_id} copy />
      <Detail label="Dataset SHA-256" value={run.dataset_sha256} copy />
      <Detail label="Evaluation protocol" value={report.evaluation_protocol} />
      <Detail label="Real rows trained" value={count(run.dataset_rows)} />
      <Detail label="Accepted batches" value={count(report.dataset_lineage?.accepted_batches?.length)} />
      <Detail label="Synthetic generated" value={count(smote.synthetic_samples_created)} />
      <Detail
        label="Synthetic in DB (this run)"
        value={
          run.persisted_synthetic_count !== undefined
            ? `${count(run.persisted_synthetic_count)} (${report.smote_database_storage?.table || 'synthetic_training_samples'})`
            : report.smote_database_storage?.verified_persisted_rows !== undefined
            ? `${count(report.smote_database_storage.verified_persisted_rows)} (${report.smote_database_storage.table || 'synthetic_training_samples'})`
            : smote.synthetic_rows_persisted_to_database !== undefined
            ? `${count(smote.synthetic_rows_persisted_to_database)} (PostgreSQL)`
            : 'None'
        }
      />
      <Detail
        label="Table total across all runs"
        value={
          run.total_stored_synthetic_count !== undefined
            ? `${count(run.total_stored_synthetic_count)} (PostgreSQL)`
            : 'Available in overview'
        }
      />
      <Detail label="Total evaluation rows" value={count(evaluationRows)} />
      <Detail label="Real rows in test" value={count(testReal)} />
      <Detail label="Synthetic rows in test" value={`${count(testSynthetic)} (SMOTENC diagnostic)`} />
      <Detail label="Champion threshold" value={comparison.champion_threshold ?? 'Unavailable'} />
      <Detail label="Challenger threshold" value={comparison.challenger_threshold ?? report.optimized?.test_metrics?.threshold ?? 'Unavailable'} />
      <Detail label="Model" value={report.optimized?.model || report.best_name || 'Unavailable'} />
      <Detail label="Training duration" value={Number.isFinite(report.training_duration_seconds) ? `${Number(report.training_duration_seconds).toFixed(1)} seconds` : 'Unavailable'} />
      <Detail label="Feature count" value={count(featureInfo.feature_count)} />
    </div>
    <h4>Feature schema</h4><p className="il-feature-list">{schema.length ? schema.join(' · ') : 'Unavailable'}</p>
    <h4>Sampler configuration</h4>
    <dl className="il-setting-list"><dt>Sampler</dt><dd>{smote.algorithm || featureInfo.sampler || 'Unavailable'}</dd>
      <dt>Sampling strategy</dt><dd>{String(settings.sampling_strategy ?? 'Unavailable')}</dd>
      <dt>Random state</dt><dd>{String(settings.random_state ?? 'Unavailable')}</dd>
      <dt>Neighbor count</dt><dd>{String(settings.k_neighbors ?? smote.k_neighbors ?? 'Unavailable')}</dd>
      <dt>Categorical positions</dt><dd>{Array.isArray(settings.categorical_feature_positions) ? settings.categorical_feature_positions.join(', ') : 'Unavailable'}</dd>
      <dt>Categorical features</dt><dd>{cats.length ? cats.join(', ') : 'Unavailable'}</dd>
      <dt>Split</dt><dd>{featureInfo.split?.method || report.split?.method || 'Unavailable'}</dd>
      <dt>Model hyperparameters</dt><dd><pre>{JSON.stringify(report.optimized?.parameters ?? {}, null, 2)}</pre></dd>
      <dt>Promotion reason</dt><dd>{report.promotion_eligibility?.reason || run.decision_reason || 'Pending'}</dd>
    </dl>
    <h4>Artifacts</h4>
    <div className="il-artifact-list">
      <Artifact label="Manifest" path={run.manifest_path} />
      <Artifact label="Candidate model" path={run.artifact_path || report.candidate_artifact} />
      {report.dataset_provenance?.resolved_path && <Artifact label="Training dataset" path={report.dataset_provenance.resolved_path} />}
    </div>
  </details>;
}

function Detail({ label, value, copy = false }: { label: string; value: unknown; copy?: boolean }) {
  return <div className="il-detail-item"><span>{label}</span><strong title={typeof value === 'string' ? value : undefined}>{value === null || value === undefined || value === '' ? 'Unavailable' : String(value)}</strong>{copy && typeof value === 'string' && <CopyButton value={value} />}</div>;
}

function Artifact({ label, path }: { label: string; path?: string }) {
  const parts = path ? path.split(/[\\/]/) : [];
  const filename = parts[parts.length - 1] || 'Unavailable';
  const shortened = parts.length > 3 ? `…\\${parts.slice(-3).join('\\')}` : path || 'Unavailable';
  return <div className="il-artifact"><span>{label}</span><strong title={path}>{filename}</strong><small title={path}>{shortened}</small>{path && <CopyButton value={path} />}</div>;
}

function CopyButton({ value }: { value: string }) {
  const [copied, setCopied] = useState(false);
  const copy = async () => {
    try { await navigator.clipboard.writeText(value); setCopied(true); window.setTimeout(() => setCopied(false), 1400); } catch { setCopied(false); }
  };
  return <button className="il-copy-button" onClick={() => void copy()}>{copied ? 'Copied' : 'Copy'}</button>;
}
