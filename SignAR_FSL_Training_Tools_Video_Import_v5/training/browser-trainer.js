(() => {
  'use strict';

  const SEQUENCE_LENGTH = 30;
  const FEATURE_DIMENSION = 456;

  function prepareDataset(samples, preferredLabels, extractor = window.SignARFeatureExtractor) {
    if (!extractor?.resampleSequence) throw new Error('The shared SignAR feature resampler is unavailable.');
    const usable = [];
    const invalidSamples = [];

    for (const sample of samples) {
      const label = String(sample?.label ?? '').trim().toUpperCase();
      if (!label) {
        invalidSamples.push({ id: sample?.id, reason: 'Missing sign label' });
        continue;
      }
      try {
        const sequence = extractor.resampleSequence(sample.features, SEQUENCE_LENGTH);
        if (sequence.some(frame => frame.length !== FEATURE_DIMENSION || frame.some(value => !Number.isFinite(value)))) {
          throw new Error('Invalid 456-feature sequence');
        }
        usable.push({
          id: sample.id,
          label,
          signerId: String(sample.signerId ?? '').trim(),
          sequence
        });
      } catch (error) {
        invalidSamples.push({ id: sample?.id, reason: error.message || 'Invalid landmark sequence' });
      }
    }

    if (!usable.length) throw new Error('No saved samples have a valid label and 456-feature sequence. Record or import a sign first.');

    const observed = new Set(usable.map(sample => sample.label));
    const labels = [...new Set([
      ...(Array.isArray(preferredLabels) ? preferredLabels : []).map(label => String(label).trim().toUpperCase()),
      ...usable.map(sample => sample.label)
    ])].filter(label => label && observed.has(label));
    const labelIndex = new Map(labels.map((label, index) => [label, index]));

    return {
      samples: usable.map(sample => ({ ...sample, classIndex: labelIndex.get(sample.label) })),
      labels,
      invalidSamples
    };
  }

  function createDatasetDocument(samples, preferredLabels = []) {
    const observedLabels = [...new Set(samples
      .map(sample => String(sample?.label ?? '').trim().toUpperCase())
      .filter(Boolean))];
    const observed = new Set(observedLabels);
    const labels = [...new Set([
      ...(Array.isArray(preferredLabels) ? preferredLabels : [])
        .map(label => String(label ?? '').trim().toUpperCase()),
      ...observedLabels
    ])].filter(label => label && observed.has(label));

    return {
      schemaVersion: 1,
      project: 'SignAR FSL Decoder',
      featureVersion: window.SignARFeatureExtractor.version,
      featureDimension: FEATURE_DIMENSION,
      samplingFps: 15,
      sequenceLength: SEQUENCE_LENGTH,
      labels,
      exportedAt: new Date().toISOString(),
      samples: samples.map(sample => ({
        id: sample.id,
        signerId: sample.signerId,
        label: sample.label,
        createdAt: new Date(sample.createdAt).toISOString(),
        verificationNotes: sample.verificationNotes || '',
        verified: !!sample.verified,
        consentRecorded: !!sample.consentRecorded,
        sourceType: sample.sourceType || 'camera',
        frameCount: sample.frameCount,
        durationMs: sample.durationMs,
        features: sample.features
      }))
    };
  }

  function splitDataset(samples, labels, seed = 42) {
    let state = seed >>> 0;
    const random = () => {
      state = (1664525 * state + 1013904223) >>> 0;
      return state / 4294967296;
    };
    const train = [];
    const validation = [];

    for (const label of labels) {
      const group = samples.filter(sample => sample.label === label);
      for (let i = group.length - 1; i > 0; i--) {
        const j = Math.floor(random() * (i + 1));
        [group[i], group[j]] = [group[j], group[i]];
      }
      const validationCount = group.length > 1 ? Math.max(1, Math.floor(group.length * 0.2)) : 0;
      train.push(...group.slice(0, group.length - validationCount));
      validation.push(...group.slice(group.length - validationCount));
    }

    return { train, validation };
  }

  function tensorData(tf, samples, singleLabel) {
    const featureValues = new Float32Array(samples.length * SEQUENCE_LENGTH * FEATURE_DIMENSION);
    let offset = 0;
    for (const sample of samples) {
      for (const frame of sample.sequence) {
        featureValues.set(frame, offset);
        offset += FEATURE_DIMENSION;
      }
    }
    const x = tf.tensor3d(featureValues, [samples.length, SEQUENCE_LENGTH, FEATURE_DIMENSION]);
    const y = singleLabel
      ? tf.tensor2d(new Float32Array(samples.length).fill(1), [samples.length, 1])
      : tf.tensor1d(Float32Array.from(samples, sample => sample.classIndex));
    return { x, y };
  }

  async function predict(model, frames, labels) {
    const tf = window.tf;
    if (!tf?.tensor3d) throw new Error('TensorFlow.js could not prepare the test sequence.');
    if (!Array.isArray(labels) || labels.length < 1) throw new Error('At least one sign label is needed for prediction.');
    const sequence = window.SignARFeatureExtractor.resampleSequence(frames, SEQUENCE_LENGTH);
    const input = tf.tensor3d(Float32Array.from(sequence.flat()), [1, SEQUENCE_LENGTH, FEATURE_DIMENSION]);
    let predictions = null;
    try {
      predictions = model.predict(input);
      if (Array.isArray(predictions)) {
        predictions.forEach(tensor => tensor.dispose());
        predictions = null;
        throw new Error('The saved model returned an unexpected set of predictions.');
      }
      const scores = Array.from(await predictions.data());
      if (scores.length !== labels.length || scores.some(score => !Number.isFinite(score))) {
        throw new Error('The model returned invalid prediction scores.');
      }
      return labels.map((label, index) => ({ label, score: scores[index] })).sort((a, b) => b.score - a.score);
    } finally {
      if (predictions) predictions.dispose();
      input.dispose();
    }
  }

  function buildModel(tf, classCount) {
    const singleLabel = classCount === 1;
    const model = tf.sequential({ name: 'signar_fsl_temporal_classifier' });
    model.add(tf.layers.gru({
      units: 96, returnSequences: true, dropout: 0.15, resetAfter: false,
      recurrentInitializer: 'glorotUniform',
      inputShape: [SEQUENCE_LENGTH, FEATURE_DIMENSION], name: 'temporal_gru_1'
    }));
    model.add(tf.layers.gru({
      units: 64, dropout: 0.20, resetAfter: false,
      recurrentInitializer: 'glorotUniform', name: 'temporal_gru_2'
    }));
    model.add(tf.layers.dense({ units: 64, activation: 'relu', name: 'representation' }));
    model.add(tf.layers.dropout({ rate: 0.25, name: 'classifier_dropout' }));
    model.add(tf.layers.dense({
      units: classCount, activation: singleLabel ? 'sigmoid' : 'softmax',
      name: 'fsl_class_probabilities'
    }));
    model.compile({
      optimizer: tf.train.adam(0.001),
      loss: singleLabel ? 'binaryCrossentropy' : 'sparseCategoricalCrossentropy',
      metrics: ['accuracy']
    });
    return model;
  }

  function evaluateValidation(probabilities, samples, labels) {
    if (!samples.length || labels.length < 2) return null;
    const confusionMatrix = labels.map(() => labels.map(() => 0));
    let correct = 0;
    samples.forEach((sample, sampleIndex) => {
      let predicted = 0;
      for (let labelIndex = 1; labelIndex < labels.length; labelIndex++) {
        if (probabilities[sampleIndex * labels.length + labelIndex] >
            probabilities[sampleIndex * labels.length + predicted]) predicted = labelIndex;
      }
      confusionMatrix[sample.classIndex][predicted]++;
      if (predicted === sample.classIndex) correct++;
    });
    return {
      accuracy: correct / samples.length,
      sampleCount: samples.length,
      confusionMatrix
    };
  }

  async function train(samples, preferredLabels, options = {}) {
    const tf = window.tf;
    if (!tf?.layers?.gru) throw new Error('TensorFlow.js could not load. Check your internet connection and refresh this page.');
    const epochs = Number(options.epochs);
    if (!Number.isInteger(epochs) || epochs < 1 || epochs > 200) throw new Error('Choose between 1 and 200 training epochs.');

    const dataset = prepareDataset(samples, preferredLabels);
    const partitions = splitDataset(dataset.samples, dataset.labels);
    if (!partitions.train.length) throw new Error('At least one usable sample is needed for training.');

    await tf.ready();
    const singleLabel = dataset.labels.length === 1;
    const trainTensors = tensorData(tf, partitions.train, singleLabel);
    const validationTensors = partitions.validation.length
      ? tensorData(tf, partitions.validation, singleLabel)
      : null;
    const model = buildModel(tf, dataset.labels.length);
    let trainingSucceeded = false;
    let finalMetrics = { accuracy: null, loss: null, validationAccuracy: null, validationLoss: null };

    try {
      const fitOptions = {
        epochs,
        batchSize: Math.min(16, partitions.train.length),
        shuffle: true,
        verbose: 0,
        callbacks: [{
          onEpochEnd: async (epoch, logs) => {
            const accuracy=Number.isFinite(logs?.accuracy)?logs.accuracy:Number.isFinite(logs?.acc)?logs.acc:null;
            if(accuracy===null)throw new Error('TensorFlow.js did not report training accuracy.');
            const validationMetric=Number.isFinite(logs?.val_accuracy)?logs.val_accuracy:Number.isFinite(logs?.val_acc)?logs.val_acc:null;
            if(validationTensors&&!singleLabel&&validationMetric===null)throw new Error('TensorFlow.js did not report validation accuracy.');
            finalMetrics = {
              accuracy,
              loss: Number.isFinite(logs?.loss) ? logs.loss : null,
              validationAccuracy:validationTensors&&!singleLabel?validationMetric:null,
              validationLoss: Number.isFinite(logs?.val_loss) ? logs.val_loss : null
            };
            if (options.onEpoch) {
              await options.onEpoch({ epoch: epoch + 1, epochs, singleLabel, ...finalMetrics });
            }
            await tf.nextFrame();
          }
        }]
      };
      if (validationTensors) fitOptions.validationData = [validationTensors.x, validationTensors.y];
      await model.fit(trainTensors.x, trainTensors.y, fitOptions);

      let evaluation = null;
      if (validationTensors && !singleLabel) {
        const predictions = model.predict(validationTensors.x);
        try {
          evaluation = evaluateValidation(
            await predictions.data(), partitions.validation, dataset.labels
          );
        } finally {
          predictions.dispose();
        }
      }

      trainingSucceeded = true;
      return {
        model,
        labels: dataset.labels,
        invalidSamples: dataset.invalidSamples,
        sampleCount: dataset.samples.length,
        trainCount: partitions.train.length,
        validationCount: partitions.validation.length,
        singleLabel,
        metrics: finalMetrics,
        evaluation
      };
    } finally {
      trainTensors.x.dispose();
      trainTensors.y.dispose();
      if (validationTensors) {
        validationTensors.x.dispose();
        validationTensors.y.dispose();
      }
      if (!trainingSucceeded) model.dispose();
    }
  }

  window.SignARBrowserTrainer = Object.freeze({ createDatasetDocument, prepareDataset, splitDataset, predict, train });
})();
