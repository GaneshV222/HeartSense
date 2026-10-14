import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';

const pagePath = new URL('../src/pages/IncrementalLearning.tsx', import.meta.url);
const resultsPath = new URL('../src/pages/IncrementalRunResults.tsx', import.meta.url);
const servicePath = new URL('../../backend/app/services/incremental_learning.py', import.meta.url);

test('run selection follows an exact requested ID and never silently falls back to the newest row', async () => {
  const page = await readFile(pagePath, 'utf8');
  const results = await readFile(resultsPath, 'utf8');

  assert.match(page, /get\('runId'\)/);
  assert.doesNotMatch(page, /if \(!selectedRunId && data\.runs\?\.length\)\s*setSelectedRunId\(data\.runs\[0\]\.training_run_id\)/);
  assert.match(page, /allRunIds=\{allRuns\.map/);
  assert.match(results, /runs\.find\(\(run\) => run\.training_run_id === selectedRunId\);/);
  assert.doesNotMatch(results, /runs\.find\([\s\S]{0,120}\)\s*\|\|\s*runs\[0\]/);
  assert.match(results, /is not present in PostgreSQL run history/);
  assert.match(results, /persisted but hidden by the rolled-back history filter/);
});

test('generated and training-partition counts keep separate provenance', async () => {
  const page = await readFile(pagePath, 'utf8');
  const results = await readFile(resultsPath, 'utf8');
  const service = await readFile(servicePath, 'utf8');

  assert.match(page, /selectedRun\?\.synthetic_samples_generated/);
  assert.match(results, /const synthCount = run\.synthetic_samples_generated;/);
  assert.match(results, /recordedCount\(selected\.synthetic_samples_generated\)/);
  assert.match(results, /synthetic_samples_used_for_training != null \? count\(selected\.synthetic_samples_used_for_training\) : 'Not recorded'/);
  assert.match(service, /"synthetic_samples_generated": expected/);
  assert.doesNotMatch(results, /synthetic_samples_generated\s*\?\?\s*(?:run\.)?(?:persisted_synthetic_count|current_run_synthetic_count)/);
});

