import { useState, useEffect } from 'react';
import { getAnalytics } from '../services/api';
import { BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

export default function Analytics() {
  const [data, setData] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchAnalytics = async () => {
      try {
        const result = await getAnalytics();
        setData(result);
      } catch (err: any) {
        setError("Error loading analytics data. Please ensure backend models are trained.");
      }
    };
    fetchAnalytics();
  }, []);

  if (error) return <div style={{ padding: '2rem', color: '#c53030', backgroundColor: '#fed7d7', margin: '2rem', borderRadius: '8px' }}>{error}</div>;
  if (!data) return <div style={{ padding: '2rem' }}>Loading analytics dashboard...</div>;

  const { dataset, smote, split, features, models, comparison, best_model, roc, confusion_matrix } = data;

  // Prepare ROC Data for Recharts
  const rocPlotData = roc.fpr.map((val: number, i: number) => ({
    fpr: Number(val.toFixed(4)),
    tpr: Number(roc.tpr[i].toFixed(4))
  }));

  // Prepare SMOTE Chart Data
  const smoteChartData = [
    { name: 'Class 0 (No Disease)', Before: smote.before['0'], After: smote.after['0'] },
    { name: 'Class 1 (Disease)', Before: smote.before['1'], After: smote.after['1'] }
  ];

  return (
    <div style={{ maxWidth: '1200px', margin: '0 auto' }}>
      <div style={{ textAlign: 'center', marginBottom: '3rem' }}>
        <h1 style={{ fontSize: '2.5rem', color: '#1a365d', marginBottom: '0.5rem' }}>HEARTSENSE ANALYTICS</h1>
        <p style={{ color: '#4a5568', fontSize: '1.2rem' }}>Machine Learning Pipeline & Evaluation Metrics</p>
      </div>

      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '3rem', padding: '1.5rem', backgroundColor: '#ebf8ff', borderRadius: '12px' }}>
        {['Data Loaded', 'Preprocessing', 'SMOTE', 'Train/Test Split', 'Feature Selection', 'Model Training', 'Evaluation'].map(step => (
          <div key={step} style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontWeight: 600, color: '#2b6cb0' }}>
            <span>✓</span> {step}
          </div>
        ))}
      </div>

      <section style={{ marginBottom: '4rem' }}>
        <h2 style={{ borderBottom: '2px solid #e2e8f0', paddingBottom: '0.5rem', color: '#2d3748' }}>Dataset Overview & Preprocessing</h2>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '1.5rem', marginTop: '1.5rem' }}>
          <div className="card" style={{ textAlign: 'center' }}>
            <h3 style={{ color: '#718096', fontSize: '1rem', margin: '0 0 0.5rem 0' }}>Raw Records</h3>
            <p style={{ fontSize: '2.5rem', fontWeight: 'bold', margin: 0, color: '#1a365d' }}>{dataset.raw_records}</p>
          </div>
          <div className="card" style={{ textAlign: 'center' }}>
            <h3 style={{ color: '#718096', fontSize: '1rem', margin: '0 0 0.5rem 0' }}>Clean Records</h3>
            <p style={{ fontSize: '2.5rem', fontWeight: 'bold', margin: 0, color: '#38a169' }}>{dataset.clean_records}</p>
          </div>
          <div className="card" style={{ textAlign: 'center' }}>
            <h3 style={{ color: '#718096', fontSize: '1rem', margin: '0 0 0.5rem 0' }}>Features</h3>
            <p style={{ fontSize: '2.5rem', fontWeight: 'bold', margin: 0, color: '#805ad5' }}>{dataset.features_raw}</p>
          </div>
          <div className="card" style={{ textAlign: 'center' }}>
            <h3 style={{ color: '#718096', fontSize: '1rem', margin: '0 0 0.5rem 0' }}>Target Variable</h3>
            <p style={{ fontSize: '1.5rem', fontWeight: 'bold', margin: '1rem 0 0 0', color: '#e53e3e' }}>Binary (0 / 1)</p>
          </div>
        </div>
      </section>

      <section style={{ marginBottom: '4rem' }}>
        <h2 style={{ borderBottom: '2px solid #e2e8f0', paddingBottom: '0.5rem', color: '#2d3748' }}>CLASS DISTRIBUTION (SMOTE)</h2>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2rem', marginTop: '1.5rem' }}>
          <div className="card">
            <h3>SMOTE Details</h3>
            <p><strong>Applied:</strong> YES</p>
            <p><strong>Method:</strong> Synthetic Minority Over-sampling Technique</p>
            <div style={{ marginTop: '1.5rem' }}>
              <p><strong>Before SMOTE:</strong></p>
              <ul>
                <li>Class 0: {smote.before['0']}</li>
                <li>Class 1: {smote.before['1']}</li>
                <li>Total: {smote.before['0'] + smote.before['1']}</li>
              </ul>
              <p><strong>After SMOTE:</strong></p>
              <ul>
                <li>Class 0: {smote.after['0']}</li>
                <li>Class 1: {smote.after['1']}</li>
                <li>Total: {smote.after['0'] + smote.after['1']}</li>
              </ul>
              <p><strong>Synthetic Samples Generated:</strong> {smote.after['1'] - smote.before['1']}</p>
            </div>
          </div>
          <div className="card">
            <h3>Class Balance (Before vs After)</h3>
            <div style={{ width: '100%', height: '300px' }}>
              <ResponsiveContainer>
                <BarChart data={smoteChartData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="Before" fill="#fc8181" />
                  <Bar dataKey="After" fill="#68d391" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      </section>

      <section style={{ marginBottom: '4rem' }}>
        <h2 style={{ borderBottom: '2px solid #e2e8f0', paddingBottom: '0.5rem', color: '#2d3748' }}>TRAIN / TEST SPLIT & FEATURE SELECTION</h2>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2rem', marginTop: '1.5rem' }}>
          <div className="card">
            <h3>Stratified Train-Test Split</h3>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem', marginTop: '1.5rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', padding: '1rem', backgroundColor: '#edf2f7', borderRadius: '8px' }}>
                <span>Total Dataset (After SMOTE)</span>
                <strong>{split.total} records</strong>
              </div>
              <div style={{ display: 'flex', gap: '1rem' }}>
                <div style={{ flex: 4, padding: '1rem', backgroundColor: '#c6f6d5', borderRadius: '8px', textAlign: 'center' }}>
                  <strong>Training Data</strong><br/>{split.train_size} records ({(1-split.test_ratio)*100}%)
                </div>
                <div style={{ flex: 1, padding: '1rem', backgroundColor: '#fed7d7', borderRadius: '8px', textAlign: 'center' }}>
                  <strong>Testing Data</strong><br/>{split.test_size} records ({split.test_ratio*100}%)
                </div>
              </div>
            </div>
          </div>
          
          <div className="card">
            <h3>Feature Selection (SelectKBest)</h3>
            <p><strong>Features Kept:</strong> {features.selected.length} / {dataset.features_raw - 1}</p>
            <div style={{ marginTop: '1rem' }}>
              <strong>Selected Features:</strong>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem', marginTop: '0.5rem' }}>
                {features.selected.map((f: string) => <span key={f} style={{ padding: '0.25rem 0.75rem', backgroundColor: '#ebf4ff', color: '#2b6cb0', borderRadius: '9999px', fontSize: '0.875rem' }}>{f}</span>)}
              </div>
            </div>
            <div style={{ marginTop: '1.5rem' }}>
              <strong>Removed Features:</strong>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem', marginTop: '0.5rem' }}>
                {features.removed.map((f: string) => <span key={f} style={{ padding: '0.25rem 0.75rem', backgroundColor: '#edf2f7', color: '#718096', borderRadius: '9999px', fontSize: '0.875rem' }}>{f}</span>)}
              </div>
            </div>
          </div>
        </div>
      </section>

      <section style={{ marginBottom: '4rem' }}>
        <h2 style={{ borderBottom: '2px solid #e2e8f0', paddingBottom: '0.5rem', color: '#2d3748' }}>MODEL COMPARISON & HYPERPARAMETER TUNING</h2>
        <div className="card" style={{ marginTop: '1.5rem', overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left' }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #e2e8f0', color: '#4a5568' }}>
                <th style={{ padding: '1rem' }}>Model</th>
                <th style={{ padding: '1rem' }}>Accuracy</th>
                <th style={{ padding: '1rem' }}>Precision</th>
                <th style={{ padding: '1rem' }}>Recall</th>
                <th style={{ padding: '1rem' }}>Specificity</th>
                <th style={{ padding: '1rem' }}>F1-Score</th>
                <th style={{ padding: '1rem' }}>ROC-AUC</th>
              </tr>
            </thead>
            <tbody>
              {comparison.map((row: any, idx: number) => (
                <tr key={idx} style={{ borderBottom: '1px solid #edf2f7', backgroundColor: row.model_name === best_model.model_name ? '#ebf8ff' : 'transparent' }}>
                  <td style={{ padding: '1rem', fontWeight: 600 }}>{row.model_name} {row.model_name === best_model.model_name && '🏆'}</td>
                  <td style={{ padding: '1rem' }}>{(row.accuracy * 100).toFixed(1)}%</td>
                  <td style={{ padding: '1rem' }}>{(row.precision * 100).toFixed(1)}%</td>
                  <td style={{ padding: '1rem' }}>{(row.recall * 100).toFixed(1)}%</td>
                  <td style={{ padding: '1rem' }}>{(row.specificity * 100).toFixed(1)}%</td>
                  <td style={{ padding: '1rem' }}>{(row.f1 * 100).toFixed(1)}%</td>
                  <td style={{ padding: '1rem', fontWeight: 'bold' }}>{row.roc_auc.toFixed(4)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div style={{ marginTop: '2rem' }}>
          <h3 style={{ marginBottom: '1.5rem' }}>Metrics Comparison Chart</h3>
          <div style={{ width: '100%', height: '400px' }}>
            <ResponsiveContainer>
              <BarChart data={comparison} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="model_name" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Bar dataKey="accuracy" name="Accuracy" fill="#8884d8" />
                <Bar dataKey="precision" name="Precision" fill="#82ca9d" />
                <Bar dataKey="recall" name="Recall" fill="#ffc658" />
                <Bar dataKey="specificity" name="Specificity" fill="#ff8042" />
                <Bar dataKey="f1" name="F1-Score" fill="#8dd1e1" />
                <Bar dataKey="roc_auc" name="ROC-AUC" fill="#a4de6c" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div style={{ marginTop: '3rem' }}>
          <details style={{ cursor: 'pointer', padding: '1rem', backgroundColor: '#f7fafc', borderRadius: '8px', border: '1px solid #e2e8f0' }}>
            <summary style={{ fontWeight: 600, color: '#4a5568' }}>Hyperparameter Tuning Details ▼</summary>
            <div style={{ marginTop: '1.5rem', display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '1.5rem' }}>
              {Object.keys(models).map(modelName => (
                <div key={modelName} className="card" style={{ padding: '1rem', border: '1px solid #e2e8f0' }}>
                  <h4 style={{ margin: '0 0 1rem 0' }}>{modelName}</h4>
                  <pre style={{ fontSize: '0.8rem', whiteSpace: 'pre-wrap', backgroundColor: '#edf2f7', padding: '0.75rem', borderRadius: '4px', margin: 0 }}>
                    {JSON.stringify(models[modelName].best_params, null, 2)}
                  </pre>
                  <p style={{ marginTop: '0.5rem', fontSize: '0.85rem' }}>CV Score: {(models[modelName].best_cv_score * 100).toFixed(2)}%</p>
                </div>
              ))}
            </div>
          </details>
        </div>
      </section>

      <section style={{ marginBottom: '4rem' }}>
        <h2 style={{ borderBottom: '2px solid #e2e8f0', paddingBottom: '0.5rem', color: '#2d3748' }}>BEST PERFORMING MODEL: {best_model.model_name}</h2>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2rem', marginTop: '1.5rem' }}>
          
          <div className="card">
            <h3 style={{ margin: '0 0 1.5rem 0' }}>ROC Curve</h3>
            <div style={{ width: '100%', height: '350px' }}>
              <ResponsiveContainer>
                <LineChart data={rocPlotData} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="fpr" type="number" label={{ value: 'False Positive Rate', position: 'bottom', offset: 0 }} />
                  <YAxis type="number" label={{ value: 'True Positive Rate', angle: -90, position: 'left' }} />
                  <Tooltip />
                  <Line type="monotone" dataKey="tpr" stroke="#e53e3e" strokeWidth={3} dot={false} name={`${best_model.model_name} (AUC = ${best_model.roc_auc.toFixed(4)})`} />
                  <Line type="monotone" dataKey="fpr" stroke="#718096" strokeWidth={1} strokeDasharray="5 5" dot={false} name="Random" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="card">
            <h3 style={{ margin: '0 0 1.5rem 0' }}>Confusion Matrix</h3>
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '350px' }}>
              <table style={{ width: '80%', borderCollapse: 'collapse', textAlign: 'center', fontSize: '1.1rem' }}>
                <thead>
                  <tr>
                    <th></th>
                    <th colSpan={2} style={{ paddingBottom: '1rem', borderBottom: '2px solid #e2e8f0' }}>Predicted</th>
                  </tr>
                  <tr>
                    <th style={{ borderRight: '2px solid #e2e8f0', paddingRight: '1rem' }}>Actual</th>
                    <th style={{ padding: '1rem' }}>Negative (0)</th>
                    <th style={{ padding: '1rem' }}>Positive (1)</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <th style={{ borderRight: '2px solid #e2e8f0', padding: '1rem' }}>Negative (0)</th>
                    <td style={{ padding: '1rem', backgroundColor: '#ebf8ff', border: '1px solid #bee3f8', fontWeight: 'bold' }}>{confusion_matrix.tn}<br/><span style={{ fontSize: '0.8rem', fontWeight: 'normal' }}>True Negative</span></td>
                    <td style={{ padding: '1rem', backgroundColor: '#fed7d7', border: '1px solid #feb2b2', fontWeight: 'bold' }}>{confusion_matrix.fp}<br/><span style={{ fontSize: '0.8rem', fontWeight: 'normal' }}>False Positive</span></td>
                  </tr>
                  <tr>
                    <th style={{ borderRight: '2px solid #e2e8f0', padding: '1rem' }}>Positive (1)</th>
                    <td style={{ padding: '1rem', backgroundColor: '#fed7d7', border: '1px solid #feb2b2', fontWeight: 'bold' }}>{confusion_matrix.fn}<br/><span style={{ fontSize: '0.8rem', fontWeight: 'normal' }}>False Negative</span></td>
                    <td style={{ padding: '1rem', backgroundColor: '#ebf8ff', border: '1px solid #bee3f8', fontWeight: 'bold' }}>{confusion_matrix.tp}<br/><span style={{ fontSize: '0.8rem', fontWeight: 'normal' }}>True Positive</span></td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>

        </div>
      </section>
      
      <section style={{ marginBottom: '4rem' }}>
        <h2 style={{ borderBottom: '2px solid #e2e8f0', paddingBottom: '0.5rem', color: '#2d3748' }}>Model Evaluation Details</h2>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '1.5rem', marginTop: '1.5rem' }}>
          <div className="card">
            <h4>Accuracy ({(best_model.accuracy * 100).toFixed(1)}%)</h4>
            <p style={{ fontSize: '0.9rem', color: '#718096' }}>How often the model makes a correct prediction overall.</p>
          </div>
          <div className="card">
            <h4>Precision ({(best_model.precision * 100).toFixed(1)}%)</h4>
            <p style={{ fontSize: '0.9rem', color: '#718096' }}>Out of all positive predictions, how many were actually positive.</p>
          </div>
          <div className="card">
            <h4>Recall / Sensitivity ({(best_model.recall * 100).toFixed(1)}%)</h4>
            <p style={{ fontSize: '0.9rem', color: '#718096' }}>How well the model identifies patients who actually have cardiovascular disease.</p>
          </div>
          <div className="card">
            <h4>Specificity ({(best_model.specificity * 100).toFixed(1)}%)</h4>
            <p style={{ fontSize: '0.9rem', color: '#718096' }}>How well the model identifies patients who do not have cardiovascular disease.</p>
          </div>
          <div className="card">
            <h4>F1-Score ({(best_model.f1 * 100).toFixed(1)}%)</h4>
            <p style={{ fontSize: '0.9rem', color: '#718096' }}>The harmonic mean of Precision and Recall.</p>
          </div>
          <div className="card">
            <h4>ROC-AUC ({best_model.roc_auc.toFixed(4)})</h4>
            <p style={{ fontSize: '0.9rem', color: '#718096' }}>The model's ability to distinguish between classes at various thresholds.</p>
          </div>
        </div>
      </section>

    </div>
  );
}
