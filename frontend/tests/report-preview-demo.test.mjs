import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';

const fixturePath = new URL('../src/pages/report-preview-demo.json', import.meta.url);
const pagePath = new URL('../src/pages/IncrementalLearning.tsx', import.meta.url);
const demoComponentPath = new URL('../src/pages/ReportPreviewDemo.tsx', import.meta.url);
const apiPath = new URL('../src/services/api.ts', import.meta.url);

test('report preview fixture contains only the requested illustrative presentation values', async () => {
  const fixture = JSON.parse(await readFile(fixturePath, 'utf8'));
  assert.deepEqual(fixture, {
    label: 'DEMO REPORT PREVIEW — ILLUSTRATIVE ONLY',
    realVisits: 109876,
    newRecords: 9875,
    syntheticGenerated: 43934,
    syntheticUsedForTraining: 43934,
    accuracyPercent: 92.95,
    recallPercent: 94.31,
    f1Percent: 93.05,
    rocAucPercent: 98.36,
    runStatus: 'Completed',
    demoDecision: 'ILLUSTRATIVE ONLY — no production decision',
  });
  for (const forbidden of ['training_run_id', 'artifact_path', 'model_version', 'promoted', 'prediction_id']) {
    assert.equal(Object.hasOwn(fixture, forbidden), false);
  }
});

test('preview is disabled by default and its component has no API client dependency', async () => {
  const page = await readFile(pagePath, 'utf8');
  const demoComponent = await readFile(demoComponentPath, 'utf8');
  const api = await readFile(apiPath, 'utf8');
  assert.match(page, /useState\(false\).*showReportPreview|showReportPreview.*useState\(false\)/s);
  assert.match(page, /showReportPreview && <ReportPreviewDemo \/>/);
  assert.match(page, /filter\(\(run\) => run\.status !== 'rolled_back'\)/);
  assert.match(page, /filter\(\(batch\) => batch\.status !== 'rolled_back'\)/);
  assert.match(demoComponent, /report-preview-demo\.json/);
  assert.doesNotMatch(demoComponent, /services\/api|fetch\s*\(|axios/);
  assert.doesNotMatch(api, /report-preview-demo|43934|92\.95/);
});
