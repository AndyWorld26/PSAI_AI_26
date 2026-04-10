import { useEffect, useState } from 'react';
import { generateData, splitData, trainFixed, trainAdaptive, runInference } from './perceptron';
import MSEChart from './MSEChart';

const N = 9;
const EPOCHS = 50000;
const PATIENCE = 100;
const TARGET_ACC = 1e-6;
const LR = 0.1;

type Results = {
  fixedTrain: number[];
  fixedTest: number[];
  adaptTrain: number[];
  adaptTest: number[];
};

export default function App() {
  const [results, setResults] = useState<Results | null>(null);

  useEffect(() => {
    const { X, Y } = generateData(N);
    const { XTrain, YTrain, XTest, YTest } = splitData(X, Y);
    const initW = Array.from({ length: N + 1 }, () => Math.random());

    const fixed = trainFixed(XTrain, YTrain, XTest, YTest, [...initW], LR, EPOCHS, PATIENCE, TARGET_ACC);
    const adaptive = trainAdaptive(XTrain, YTrain, XTest, YTest, [...initW], EPOCHS, PATIENCE, TARGET_ACC);

    const { weights, threshold } = adaptive;

    (window as unknown as Record<string, unknown>).perceptron = {
      run: (input: number[]) => {
        const result = runInference(input, weights, threshold);
        console.log(`Input:       [${input.join(', ')}]`);
        console.log(`Probability: ${result.prob.toFixed(6)}`);
        console.log(`Class:       ${result.cls}`);
        return result;
      },
      runAll: () => {
        const { X: allX, Y: allY } = generateData(N);
        console.table(
          allX.map((row, i) => ({
            input: `[${row.join(',')}]`,
            expected: allY[i],
            probability: runInference(row, weights, threshold).prob.toFixed(6),
            predicted: runInference(row, weights, threshold).cls,
            correct: runInference(row, weights, threshold).cls === allY[i] ? '✓' : '✗',
          }))
        );
      },
      weights,
      threshold,
    };

    console.log('%cPerceptron ready', 'color: #52e08a; font-weight: bold');
    console.log('perceptron.run([0,1,1,1,1,1,1,1,1])  — classify one vector');
    console.log('perceptron.runAll()                   — print full truth table');
    console.log('perceptron.weights                    — adaptive LR weights');
    console.log('perceptron.threshold                  — threshold value');

    setResults({
      fixedTrain: fixed.trainHistory,
      fixedTest: fixed.testHistory,
      adaptTrain: adaptive.trainHistory,
      adaptTest: adaptive.testHistory,
    });
  }, []);

  if (!results) return <p>Training...</p>;

  return (
    <MSEChart
      fixedTrain={results.fixedTrain}
      fixedTest={results.fixedTest}
      adaptTrain={results.adaptTrain}
      adaptTest={results.adaptTest}
    />
  );
}
