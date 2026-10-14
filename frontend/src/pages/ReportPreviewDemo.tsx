import demo from './report-preview-demo.json';

const demoCards: [string, number | string][] = [
  ['Real visits', demo.realVisits],
  ['New records', `+${demo.newRecords.toLocaleString()}`],
  ['Synthetic generated in run', demo.syntheticGenerated],
  ['Synthetic used in training', demo.syntheticUsedForTraining],
  ['Accuracy', `${demo.accuracyPercent.toFixed(2)}%`],
  ['Recall', `${demo.recallPercent.toFixed(2)}%`],
  ['F1', `${demo.f1Percent.toFixed(2)}%`],
  ['ROC-AUC', `${demo.rocAucPercent.toFixed(2)}%`],
  ['Run status', demo.runStatus],
  ['Demo decision', demo.demoDecision],
];

export default function ReportPreviewDemo() {
  return <div
    role="note"
    aria-label={demo.label}
    style={{ marginTop: 14, padding: 14, border: '2px dashed #d97706', borderRadius: 10, background: '#fffbeb' }}
  >
    <strong style={{ color: '#92400e' }}>{demo.label}</strong>
    <p style={{ color: '#78350f', margin: '6px 0 12px' }}>
      Presentation fixture only. These values are not a persisted training run, evaluation, or model decision.
    </p>
    <div className="il-summary-grid">
      {demoCards.map(([label, value]) => <div className="il-summary-item" key={label}>
        <span>{label}</span><strong>{typeof value === 'number' ? value.toLocaleString() : value}</strong>
      </div>)}
    </div>
  </div>;
}
