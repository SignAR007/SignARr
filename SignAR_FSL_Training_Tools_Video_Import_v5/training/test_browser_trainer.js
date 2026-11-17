/* Node.js tests for browser training data preparation; no TensorFlow.js dependency required. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

global.window = {};
vm.runInThisContext(fs.readFileSync(path.join(__dirname, '..', 'shared', 'fsl-features.js'), 'utf8'));
vm.runInThisContext(fs.readFileSync(path.join(__dirname, 'browser-trainer.js'), 'utf8'));

const trainer = global.window.SignARBrowserTrainer;
const frames = [new Array(456).fill(0), new Array(456).fill(1)];
const datasetDocument = trainer.createDatasetDocument([
  { id: 'existing', label: 'KAMUSTA KA', createdAt: 0 },
  { id: 'new-a', label: 'A', createdAt: 1 },
  { id: 'new-b', label: 'I love you', createdAt: 2 },
  { id: 'duplicate', label: 'a', createdAt: 3 }
], ['KAMUSTA KA', 'UNUSED OLD LABEL']);
assert.deepEqual(datasetDocument.labels, ['KAMUSTA KA', 'A', 'I LOVE YOU'], 'export every observed label without stale unused vocabulary');
assert.equal(datasetDocument.samples.length, 4, 'preserve all saved samples in the export');
const samples = [
  { id: 'one-a', label: 'word a', signerId: 'signer-01', features: frames },
  { id: 'two-a', label: 'WORD A', signerId: 'signer-02', features: frames },
  { id: 'one-b', label: 'word b', signerId: 'signer-01', features: frames },
  { id: 'two-b', label: 'WORD B', signerId: 'signer-02', features: frames },
  { id: 'bad', label: 'WORD A', features: [[0, 1]] }
];

const prepared = trainer.prepareDataset(samples, ['WORD B', 'WORD A']);
assert.deepEqual(prepared.labels, ['WORD B', 'WORD A'], 'preserve preferred label order for observed classes');
assert.equal(prepared.samples.length, 4, 'skip invalid feature sequences');
assert.equal(prepared.samples[0].sequence.length, 30, 'resample each clip to the model window');
assert.equal(prepared.samples[0].sequence[0].length, 456, 'preserve the feature width');
assert.equal(prepared.invalidSamples.length, 1, 'report invalid samples');

const split = trainer.splitDataset(prepared.samples, prepared.labels);
assert.equal(split.train.length + split.validation.length, prepared.samples.length, 'retain all valid samples');
assert.equal(split.train.length, 2, 'keep one training example for each class');
assert.equal(split.validation.length, 2, 'reserve held-out examples when available');
assert.deepEqual(trainer.splitDataset(prepared.samples, prepared.labels), split, 'produce a repeatable sample-level split');

assert.throws(
  () => trainer.prepareDataset([{ id: 'empty', label: 'WORD A', features: [] }], []),
  /No saved samples have a valid label/
);
console.log('Browser trainer data-preparation tests passed.');

(async () => {
  let inputDisposed = false;
  let predictionDisposed = false;
  global.window.tf = {
    tensor3d(values, shape) {
      assert.equal(values.length, 30 * 456, 'resample a test sign to the model input size');
      assert.deepEqual(shape, [1, 30, 456], 'include a single sequence batch');
      return { dispose() { inputDisposed = true; } };
    }
  };
  const predictions = await trainer.predict({
    predict() {
      return {
        async data() { return new Float32Array([0.2, 0.8]); },
        dispose() { predictionDisposed = true; }
      };
    }
  }, frames, ['WORD A', 'WORD B']);
  assert.deepEqual(predictions.map(result => result.label), ['WORD B', 'WORD A'], 'rank predictions by model score');
  assert.ok(Math.abs(predictions[0].score - 0.8) < 1e-6);
  assert.ok(Math.abs(predictions[1].score - 0.2) < 1e-6);
  assert.equal(inputDisposed, true, 'dispose the prepared input tensor');
  assert.equal(predictionDisposed, true, 'dispose the prediction tensor');
  const singleLabelPrediction = await trainer.predict({
    predict() {
      return {
        async data() { return new Float32Array([0.91]); },
        dispose() {}
      };
    }
  }, frames, ['WORD A']);
  assert.deepEqual(singleLabelPrediction.map(result => result.label), ['WORD A'], 'allow prediction with one vocabulary label');
  assert.ok(Math.abs(singleLabelPrediction[0].score - 0.91) < 1e-6);
  console.log('Browser trainer prediction tests passed.');

  let compileMetrics = null;
  let trainingPredictionCount = 0;
  const tensor = { dispose() {} };
  const trainingModel = {
    add() {},
    compile(options) { compileMetrics = options.metrics; },
    async fit(_x, _y, options) {
      await options.callbacks[0].onEpochEnd(0, {
        loss: 0.4, acc: 0.75, val_loss: 0.5, val_acc: 0.5
      });
    },
    predict() {
      trainingPredictionCount++;
      return {
        async data() { return new Float32Array([0.8, 0.2, 0.2, 0.8]); },
        dispose() {}
      };
    }
  };
  global.window.tf = {
    layers: { gru: () => ({}), dense: () => ({}), dropout: () => ({}) },
    train: { adam: () => ({}) },
    sequential: () => trainingModel,
    async ready() {},
    tensor3d() { return tensor; },
    tensor1d() { return tensor; },
    async nextFrame() {}
  };
  const trained = await trainer.train(samples.slice(0, 4), ['WORD A', 'WORD B'], {
    epochs: 1
  });
  assert.deepEqual(compileMetrics, ['accuracy'], 'compute accuracy as part of model.fit');
  assert.equal(trainingPredictionCount, 1, 'only evaluate validation once after training, not again each epoch');
  assert.equal(trained.metrics.accuracy, 0.75, 'use fit training accuracy logs');
  assert.equal(trained.metrics.validationAccuracy, 0.5, 'use fit validation accuracy logs');
  assert.equal(trained.evaluation.accuracy, 1, 'preserve final validation evaluation');
  console.log('Browser trainer avoids redundant per-epoch predictions.');
})().catch(error => {
  console.error(error);
  process.exitCode = 1;
});
