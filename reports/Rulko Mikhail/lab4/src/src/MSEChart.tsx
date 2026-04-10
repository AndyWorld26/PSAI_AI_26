import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';

type Props = {
  fixedTrain: number[];
  fixedTest: number[];
  adaptTrain: number[];
  adaptTest: number[];
};

export default function MSEChart({ fixedTrain, fixedTest, adaptTrain, adaptTest }: Props) {
  const length = Math.max(fixedTrain.length, adaptTrain.length);
  const data = Array.from({ length }, (_, i) => ({
    epoch: i,
    fixedTrain: fixedTrain[i] ?? null,
    fixedTest: fixedTest[i] ?? null,
    adaptTrain: adaptTrain[i] ?? null,
    adaptTest: adaptTest[i] ?? null,
  }));

  return (
    <ResponsiveContainer width="100%" height={380}>
      <LineChart data={data} margin={{ top: 8, right: 24, left: 8, bottom: 8 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#2a2a2a" />
        <XAxis
          dataKey="epoch"
          label={{ value: 'Epoch', position: 'insideBottom', offset: -4, fill: '#888' }}
          tick={{ fill: '#888', fontSize: 11 }}
        />
        <YAxis
          label={{ value: 'MSE', angle: -90, position: 'insideLeft', fill: '#888' }}
          tick={{ fill: '#888', fontSize: 11 }}
          tickFormatter={(v: number) => v.toFixed(4)}
        />
        <Tooltip
          contentStyle={{ background: '#111', border: '1px solid #333', fontSize: 12 }}
          formatter={(v: number) => v?.toFixed(6)}
        />
        <Legend wrapperStyle={{ fontSize: 12, paddingTop: 12 }} />
        <Line type="monotone" dataKey="fixedTrain" name="Fixed LR Train" stroke="#e05252" dot={false} strokeWidth={1.5} strokeDasharray="5 3" />
        <Line type="monotone" dataKey="fixedTest" name="Fixed LR Test" stroke="#e05252" dot={false} strokeWidth={1.5} />
        <Line type="monotone" dataKey="adaptTrain" name="Adaptive LR Train" stroke="#52a8e0" dot={false} strokeWidth={1.5} strokeDasharray="5 3" />
        <Line type="monotone" dataKey="adaptTest" name="Adaptive LR Test" stroke="#52a8e0" dot={false} strokeWidth={1.5} />
      </LineChart>
    </ResponsiveContainer>
  );
}
