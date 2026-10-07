import { useEffect, useMemo, useState } from 'react';
import { getAnalytics } from '../services/api';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import {
  Activity,
  Database,
  ShieldCheck,
  TrendingUp,
} from 'lucide-react';

const metricKeys = [
  'accuracy',
  'precision',
  'recall',
  'specificity',
  'f1',
  'roc_auc',
  'pr_auc',
];

const metricLabels: Record<string, string> = {
  accuracy: 'Accuracy',
  precision: 'Precision',
  recall: 'Recall',
  specificity: 'Specificity',
  f1: 'F1 Score',
  roc_auc: 'ROC-AUC',
  pr_auc: 'PR-AUC',
};

const pctMetrics = new Set(['accuracy', 'precision', 'recall', 'specificity', 'f1']);

const formatMetric = (value: any, key: string) => {
  const n = Number(value);
  if (!Number.isFinite(n)) return '-';
  if (pctMetrics.has(key)) return `${(n * 100).toFixed(2)}%`;
  return n.toFixed(4);
};

const formatPct = (value: any) => {
  const n = Number(value);
  return Number.isFinite(n) ? `${(n * 100).toFixed(2)}%` : '-';
};

const getModelName = (row: any) => row?.model_name || row?.name || '-';

export default function Analytics() {
  const [data, setData] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchAnalytics = async () => {
      try {
        setData(await getAnalytics());
      } catch {
        setError('Unable to load analytics. Train the backend model once, then refresh this page.');
      }
    };
    fetchAnalytics();
  }, []);

  const derived = useMemo(() => {
    if (!data) return null;

    const holdoutRows = Array.isArray(data.comparative_analysis) ? data.comparative_analysis : [];
    const afterRows = Array.isArray(data.comparison) ? data.comparison : [];
    const beforeMetricRows = Array.isArray(data.before_temporal?.models) ? data.before_temporal.models : [];
    const afterMetricRows = Array.isArray(data.after_temporal?.models) ? data.after_temporal.models : afterRows;
    const bestAfter =
      afterRows.find((row: any) => getModelName(row) === data.best_model?.model_name) ||
      afterRows[0] ||
      data.best_model ||
      {};

    const bestHoldout =
      holdoutRows.find((row: any) => getModelName(row) === getModelName(bestAfter)) ||
      holdoutRows[0] ||
      {};

    return {
      holdoutRows,
      beforeMetricRows,
      afterMetricRows,
      bestAfter,
      bestHoldout,
    };
  }, [data]);

  if (error) {
    return <MessageBox tone="error" title="Analytics Error" message={error} />;
  }

  if (!data || !derived) {
    return <MessageBox tone="neutral" title="Loading Model Analytics" message="Reading evaluation metrics, SMOTE results, and temporal comparison artifacts." />;
  }

  const { smote_analysis, confusion_matrix } = data;
  const {
    holdoutRows,
    beforeMetricRows,
    afterMetricRows,
    bestAfter,
    bestHoldout,
  } = derived;

  const smoteBefore = smote_analysis?.distribution_before || {};
  const smoteAfter = smote_analysis?.distribution_after || {};
  const smoteChartData = [
    { name: 'Class 0', Before: smoteBefore['0'] || smoteBefore[0] || 0, After: smoteAfter['0'] || smoteAfter[0] || 0 },
    { name: 'Class 1', Before: smoteBefore['1'] || smoteBefore[1] || 0, After: smoteAfter['1'] || smoteAfter[1] || 0 },
  ];

  const cm = confusion_matrix || bestAfter?.confusion_matrix || {};
  const tn = Number(cm.tn || 0);
  const fp = Number(cm.fp || 0);
  const fn = Number(cm.fn || 0);
  const tp = Number(cm.tp || 0);

  return (
    <div style={{ maxWidth: '1200px', margin: '0 auto', padding: '1rem 0 3rem' }}>
      <header style={{ marginBottom: '1.5rem' }}>
        <h1 style={{ color: '#102a43', fontSize: '2rem', margin: 0, fontWeight: 800 }}>
          Model Evaluation Analytics
        </h1>
      </header>

      <section style={{ display: 'grid', gridTemplateColumns: 'repeat(2, minmax(0, 1fr))', gap: '1rem', marginBottom: '1.25rem' }}>
        <KpiCard title="Best Model" value={getModelName(bestAfter)} sub="Selected from after SMOTE + temporal pipeline" icon={<ShieldCheck size={19} />} />
        <KpiCard title="Final Holdout Accuracy" value={formatPct(bestAfter.accuracy ?? bestHoldout.after_accuracy)} sub="After SMOTE + temporal data" icon={<TrendingUp size={19} />} />
      </section>

      <SectionTitle icon={<Activity size={20} />} title="1. Baseline Model Evaluation Before SMOTE And Temporal Data" />
      <div style={panelStyle}>
        <MetricsTable
          rows={beforeMetricRows.length ? beforeMetricRows : holdoutRows}
          emptyText="Baseline before/after comparison artifact is missing. Run backend/train.py to generate before_after_temporal_comparison.csv."
          columns={[
            { key: 'model_name', label: 'Model' },
            { key: 'accuracy', fallbackKey: 'before_accuracy', label: 'Accuracy', metric: 'accuracy' },
            { key: 'precision', label: 'Precision', metric: 'precision' },
            { key: 'recall', fallbackKey: 'before_recall', label: 'Recall', metric: 'recall' },
            { key: 'specificity', label: 'Specificity', metric: 'specificity' },
            { key: 'f1', fallbackKey: 'before_f1', label: 'F1', metric: 'f1' },
            { key: 'roc_auc', fallbackKey: 'before_roc_auc', label: 'ROC-AUC', metric: 'roc_auc' },
            { key: 'pr_auc', label: 'PR-AUC', metric: 'pr_auc' },
          ]}
        />
      </div>

      <SectionTitle icon={<TrendingUp size={20} />} title="2. Model Accuracy After SMOTE And Temporal Data" />
      <div style={panelStyle}>
        <MetricsTable
          rows={afterMetricRows.length ? afterMetricRows : holdoutRows}
          emptyText="After-model comparison artifact is missing."
          columns={[
            { key: 'model_name', label: 'Model' },
            { key: 'after_accuracy', fallbackKey: 'accuracy', label: 'Accuracy', metric: 'accuracy' },
            { key: 'precision', label: 'Precision', metric: 'precision' },
            { key: 'specificity', label: 'Specificity', metric: 'specificity' },
            { key: 'after_f1', fallbackKey: 'f1', label: 'F1', metric: 'f1' },
            { key: 'after_recall', fallbackKey: 'recall', label: 'Recall', metric: 'recall' },
            { key: 'after_roc_auc', fallbackKey: 'roc_auc', label: 'ROC-AUC', metric: 'roc_auc' },
            { key: 'pr_auc', label: 'PR-AUC', metric: 'pr_auc' },
          ]}
        />
      </div>

      <div style={{ marginTop: '1rem', marginBottom: '1.25rem' }}>
        <div style={{ ...panelStyle, minWidth: 0 }}>
          <h3 style={smallHeading}>SMOTE Class Balance</h3>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.75rem', marginBottom: '0.75rem' }}>
            <MetricCard label="Before SMOTE" value={`${Number(smote_analysis?.records_before || 0).toLocaleString()} rows`} />
            <MetricCard label="After SMOTE" value={`${Number(smote_analysis?.records_after || 0).toLocaleString()} rows`} />
          </div>
          <div style={{ height: 260, width: '100%' }}>
            <ResponsiveContainer>
              <BarChart data={smoteChartData} margin={{ top: 12, right: 12, left: 4, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis width={58} tickFormatter={(value) => Number(value).toLocaleString()} />
                <Tooltip />
                <Legend wrapperStyle={{ fontSize: '0.8rem' }} />
                <Bar dataKey="Before" fill="#ef4444" name="Before SMOTE" />
                <Bar dataKey="After" fill="#16a34a" name="After SMOTE" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      <SectionTitle icon={<Database size={20} />} title="Complete Evaluation Metrics For Final Model" />
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginBottom: '1.25rem' }}>
        <div style={panelStyle}>
          <h3 style={smallHeading}>{getModelName(bestAfter)} Metrics</h3>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '0.75rem' }}>
            {metricKeys.map((key) => (
              <MetricCard key={key} label={metricLabels[key]} value={formatMetric(bestAfter[key], key)} />
            ))}
          </div>
        </div>
        <div style={panelStyle}>
          <h3 style={smallHeading}>Confusion Matrix</h3>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.5rem', textAlign: 'center' }}>
            <MatrixCell label="True Negative" value={tn} color="#1d4ed8" bg="#eff6ff" />
            <MatrixCell label="False Positive" value={fp} color="#b91c1c" bg="#fef2f2" />
            <MatrixCell label="False Negative" value={fn} color="#b91c1c" bg="#fef2f2" />
            <MatrixCell label="True Positive" value={tp} color="#15803d" bg="#f0fdf4" />
          </div>
        </div>
      </div>
    </div>
  );
}

function SectionTitle({ title, icon }: { title: string; icon: any }) {
  return (
    <h2 style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', color: '#243b53', fontSize: '1.18rem', margin: '1.5rem 0 0.75rem' }}>
      {icon}
      {title}
    </h2>
  );
}

function MetricsTable({ rows, columns, emptyText }: any) {
  if (!rows || rows.length === 0) {
    return <p style={{ margin: 0, color: '#64748b' }}>{emptyText}</p>;
  }

  return (
    <div style={{ overflowX: 'auto' }}>
      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.88rem' }}>
        <thead>
          <tr style={{ background: '#f8fafc', color: '#334155', borderBottom: '1px solid #d9e2ec' }}>
            {columns.map((col: any) => (
              <th key={col.key} style={{ padding: '0.75rem', textAlign: col.key === 'model_name' ? 'left' : 'center', whiteSpace: 'nowrap' }}>
                {col.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row: any, index: number) => (
            <tr key={`${getModelName(row)}-${index}`} style={{ borderBottom: '1px solid #edf2f7' }}>
              {columns.map((col: any) => {
                const raw = row[col.key] ?? row[col.fallbackKey];
                const value =
                  col.key === 'model_name'
                    ? getModelName(row)
                    : typeof raw === 'boolean'
                      ? raw ? 'Yes' : 'No'
                      : col.metric
                        ? formatMetric(raw, col.metric)
                        : raw ?? '-';
                return (
                  <td key={col.key} style={{ padding: '0.75rem', textAlign: col.key === 'model_name' ? 'left' : 'center', fontWeight: col.key === 'model_name' ? 700 : 600, color: '#1f2937', whiteSpace: 'nowrap' }}>
                    {value}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function KpiCard({ title, value, sub, icon, tone = 'neutral' }: any) {
  const colors: any = {
    neutral: ['#e0f2fe', '#075985'],
    success: ['#dcfce7', '#166534'],
    warning: ['#fef3c7', '#92400e'],
  };
  const [bg, fg] = colors[tone] || colors.neutral;
  return (
    <div style={{ ...panelStyle, padding: '1rem' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', color: fg }}>
        <span style={{ fontSize: '0.78rem', fontWeight: 800, textTransform: 'uppercase' }}>{title}</span>
        <span style={{ background: bg, borderRadius: 8, padding: '0.35rem', display: 'inline-flex' }}>{icon}</span>
      </div>
      <div style={{ marginTop: '0.65rem', fontSize: '1.35rem', fontWeight: 850, color: '#102a43', lineHeight: 1.15 }}>{value}</div>
      <div style={{ marginTop: '0.25rem', color: '#64748b', fontSize: '0.8rem' }}>{sub}</div>
    </div>
  );
}

function MetricCard({ label, value }: any) {
  return (
    <div style={{ background: '#f8fafc', border: '1px solid #e2e8f0', borderRadius: 8, padding: '0.75rem' }}>
      <div style={{ color: '#64748b', fontSize: '0.75rem', fontWeight: 800, textTransform: 'uppercase' }}>{label}</div>
      <div style={{ color: '#0f172a', fontWeight: 850, fontSize: '1.05rem', marginTop: '0.25rem' }}>{value}</div>
    </div>
  );
}

function MatrixCell({ label, value, color, bg }: any) {
  return (
    <div style={{ background: bg, border: '1px solid #e2e8f0', borderRadius: 8, padding: '1rem' }}>
      <div style={{ color, fontWeight: 850, fontSize: '1.35rem' }}>{Number(value || 0).toLocaleString()}</div>
      <div style={{ color: '#64748b', fontSize: '0.78rem', fontWeight: 700 }}>{label}</div>
    </div>
  );
}

function MessageBox({ title, message, tone }: any) {
  const isError = tone === 'error';
  return (
    <div style={{ padding: '2rem', color: isError ? '#991b1b' : '#334155', backgroundColor: isError ? '#fee2e2' : '#f8fafc', margin: '2rem auto', maxWidth: '900px', borderRadius: '8px', border: '1px solid #e2e8f0' }}>
      <strong>{title}</strong>
      <p style={{ marginBottom: 0 }}>{message}</p>
    </div>
  );
}

const panelStyle: React.CSSProperties = {
  background: '#ffffff',
  border: '1px solid #d9e2ec',
  borderRadius: 8,
  boxShadow: '0 1px 3px rgba(15, 23, 42, 0.06)',
  padding: '1rem',
};

const smallHeading: React.CSSProperties = {
  color: '#102a43',
  fontSize: '1rem',
  fontWeight: 800,
  margin: '0 0 0.85rem',
};
