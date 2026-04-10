export type TrainResult = 
{

    trainHistory: number[];
    testHistory: number[];
    weights: number[];
    threshold: number;
    epochs: number;

};

  function dot(a: number[], b: number[]): number 
  {
    return a.reduce((s, v, i) => s + v * b[i], 0);
  }

  function sigmoid(x: number): number 
  {
    return 1 / (1 + Math.exp(-x));
  }

  function insertBias(X: number[][]): number[][] 
  {
    return X.map(row => [-1, ...row]);
  }

  function mse(y: number[], target: number[]): number 
  {
    return y.reduce((s, v, i) => s + (v - target[i]) ** 2, 0) / y.length;
  }

  function predict(X: number[][], w: number[]): number[] 
  {
    return X.map(row => sigmoid(dot(row, w)));
  }

function delta(
  X: number[][],
  y: number[],
  target: number[],
  w: number[],
  lr: number
): number[] {
  const grad = new Array(w.length).fill(0);
  for (let i = 0; i < y.length; i++) {
    const err = y[i] - target[i];
    const der = y[i] * (1 - y[i]);
    for (let j = 0; j < w.length; j++) {
      grad[j] += err * der * X[i][j];
    }
  }
  return w.map((wj, j) => wj - lr * (grad[j] / y.length));
}

function adaptiveLR(X: number[][]): number {
  const sumSq = X.reduce((s, row) => s + row.reduce((rs, v) => rs + v * v, 0), 0);
  return 1 / (sumSq / X.length);
}

export function generateData(N: number): { X: number[][]; Y: number[] } {
  const total = 1 << N;
  const X: number[][] = [];
  const Y: number[] = [];
  const inverseMask = [1, 0, 0, 0, 0, 0, 0, 0, 0].slice(0, N);

  for (let i = 0; i < total; i++) {
    const row: number[] = [];
    for (let b = N - 1; b >= 0; b--) {
      row.push((i >> b) & 1);
    }
    X.push(row);

    const modified = row.map((v, j) => (inverseMask[j] === 1 ? 1 - v : v));
    const target = modified.reduce((acc, v) => acc & v, 1);
    Y.push(target);
  }

  return { X, Y };
}

export function splitData(
  X: number[][],
  Y: number[],
  trainRatio = 0.9
): { XTrain: number[][]; YTrain: number[]; XTest: number[][]; YTest: number[] } {
  const onesIdx = Y.map((v, i) => (v === 1 ? i : -1)).filter(i => i >= 0);
  const zerosIdx = Y.map((v, i) => (v === 0 ? i : -1)).filter(i => i >= 0);

  const zerosShuffled = [...zerosIdx].sort(() => Math.random() - 0.5);
  const split = Math.floor(trainRatio * zerosShuffled.length);
  const trainIdx = [...onesIdx, ...zerosShuffled.slice(0, split)];
  const testIdx = zerosShuffled.slice(split);

  return {
    XTrain: trainIdx.map(i => X[i]),
    YTrain: trainIdx.map(i => Y[i]),
    XTest: testIdx.map(i => X[i]),
    YTest: testIdx.map(i => Y[i]),
  };
}

export function trainFixed(
  XTrain: number[][],
  YTrain: number[],
  XTest: number[][],
  YTest: number[],
  initialWeights: number[],
  lr: number,
  epochs: number,
  patience: number,
  targetAccuracy: number
): 
TrainResult 
{
  const XBias = insertBias(XTrain);
  const XTestBias = insertBias(XTest);
  let w = [...initialWeights];
  const trainHistory: number[] = [];
  const testHistory: number[] = [];
  let lastEpoch = epochs;

  for (let epoch = 0; epoch < epochs; epoch++) {
    const y = predict(XBias, w);
    const err = mse(y, YTrain);
    trainHistory.push(err);
    testHistory.push(mse(predict(XTestBias, w), YTest));

    if (err <= targetAccuracy) {
      lastEpoch = epoch + 1;
      break;
    }

    if (trainHistory.length > patience) {
      const recent = trainHistory.slice(-patience);
      if (Math.max(...recent) - Math.min(...recent) < targetAccuracy) {
        lastEpoch = epoch + 1;
        break;
      }
    }

    w = delta(XBias, y, YTrain, w, lr);
  }

  return {
    trainHistory,
    testHistory,
    weights: w.slice(1),
    threshold: w[0],
    epochs: lastEpoch,
  };
}

export function trainAdaptive(
  XTrain: number[][],
  YTrain: number[],
  XTest: number[][],
  YTest: number[],
  initialWeights: number[],
  epochs: number,
  patience: number,
  targetAccuracy: number
): 
TrainResult 
{
  const XBias = insertBias(XTrain);
  const XTestBias = insertBias(XTest);
  let w = [...initialWeights];
  const lr = adaptiveLR(XBias);
  const trainHistory: number[] = [];
  const testHistory: number[] = [];
  let lastEpoch = epochs;

  for (let epoch = 0; epoch < epochs; epoch++) {
    const y = predict(XBias, w);
    const err = mse(y, YTrain);
    trainHistory.push(err);
    testHistory.push(mse(predict(XTestBias, w), YTest));

    if (err <= targetAccuracy) {
      lastEpoch = epoch + 1;
      break;
    }

    if (trainHistory.length > patience) {
      const recent = trainHistory.slice(-patience);
      if (Math.max(...recent) - Math.min(...recent) < targetAccuracy) {
        lastEpoch = epoch + 1;
        break;
      }
    }

    w = delta(XBias, y, YTrain, w, lr);
  }

  return {
    trainHistory,
    testHistory,
    weights: w.slice(1),
    threshold: w[0],
    epochs: lastEpoch,
  };
}

export function runInference(input: number[], weights: number[], threshold: number): { prob: number; cls: number } 
{
  const w = [threshold, ...weights];
  const x = [-1, ...input];
  const prob = sigmoid(dot(x, w));
  return { prob, cls: prob > 0.5 ? 1 : 0 };
}
