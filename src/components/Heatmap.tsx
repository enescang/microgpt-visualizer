import { Value } from "../engine/autograd";

const DISPLAY_DECIMALS = 2;
const TOOLTIP_DECIMALS = 4;

function valToColor(v: number, scale = 0.3): string {
  const t = Math.max(-1, Math.min(1, v / scale));
  if (t < 0) {
    // warm terracotta for negatives
    const r = Math.floor(140 - t * 95);
    const g = Math.floor(120 + t * 50);
    const b = Math.floor(110 + t * 50);
    return `rgb(${r},${g},${b})`;
  } else {
    // sage green for positives
    const r = Math.floor(130 + t * 10);
    const g = Math.floor(135 + t * 80);
    const b = Math.floor(120 + t * 20);
    return `rgb(${r},${g},${b})`;
  }
}

interface Props {
  matrix: Value[][];
  rowLabels: string[];
  colCount: number;
  highlightRow?: number;
  onHoverRow?: (row: number | null) => void;
}

export default function Heatmap({ matrix, rowLabels, colCount, highlightRow, onHoverRow }: Props) {
  return (
    <div className="heatmap-wrap">
      <table className="heatmap">
        <thead>
          <tr>
            <th></th>
            {Array.from({ length: colCount }, (_, c) => (
              <th key={c}>d{c}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {matrix.map((row, r) => (
            <tr
              key={r}
              onMouseEnter={() => onHoverRow?.(r)}
              onMouseLeave={() => onHoverRow?.(null)}
              style={highlightRow === r ? { outline: "2px solid var(--blue)" } : undefined}
            >
              <td className="row-label">{rowLabels[r]}</td>
              {row.slice(0, colCount).map((cell, c) => {
                const v = cell.data;
                const bg = valToColor(v);
                return (
                  <td
                    key={c}
                    style={{ background: bg, color: Math.abs(v) > 0.25 ? "#2a2a25" : "#4a4a42" }}
                    title={`${rowLabels[r]} dim${c}: ${v.toFixed(TOOLTIP_DECIMALS)}`}
                  >
                    {v.toFixed(DISPLAY_DECIMALS)}
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

export function VectorBar({ values, label }: { values: number[]; label?: string }) {
  const maxAbs = Math.max(...values.map(Math.abs), 0.01);
  return (
    <div>
      {label && <div style={{ fontSize: 10, color: "var(--text-dim)", marginBottom: 2 }}>{label}</div>}
      <div className="vector-display">
        {values.map((v, i) => (
          <div
            key={i}
            className="vector-cell"
            style={{ background: valToColor(v, maxAbs * 0.8) }}
            title={`dim${i}: ${v.toFixed(TOOLTIP_DECIMALS)}`}
          >
            {v.toFixed(DISPLAY_DECIMALS)}
          </div>
        ))}
      </div>
    </div>
  );
}
