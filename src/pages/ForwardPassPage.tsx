import { useState, useMemo } from "react";
import { type ModelState, gptForward, uchars, charToId, tokenLabel, N_LAYER, N_HEAD } from "../engine/model";
import { VectorBar } from "../components/Heatmap";

interface Props {
  model: ModelState;
}

export default function ForwardPassPage({ model }: Props) {
  const [char, setChar] = useState("e");
  const [pos, setPos] = useState(0);

  const tokenId = charToId[char] ?? 0;
  const trace = useMemo(() => {
    const keys = Array.from({ length: N_LAYER }, () => [] as any[]);
    const vals = Array.from({ length: N_LAYER }, () => [] as any[]);
    const result = gptForward(tokenId, pos, keys, vals, model, true);
    return result.trace!;
  }, [model, char, pos, model.totalStep]);

  const top5 = trace.probs
    .map((p, i) => ({ id: i, char: tokenLabel(i), prob: p }))
    .sort((a, b) => b.prob - a.prob)
    .slice(0, 10);
  const maxProb = Math.max(...top5.map((t) => t.prob), 0.01);

  return (
    <>
      <h1 className="page-title">3. Forward Pass</h1>
      <p className="page-desc">
        Watch one token flow through the entire model. Each step transforms the 16-number
        vector until it becomes 27 probability scores — one for each possible next character.
      </p>

      {/* Controls */}
      <div className="panel">
        <div className="panel-title">Select Input</div>
        <div className="controls">
          <span style={{ fontSize: 12, color: "var(--text-dim)" }}>Token:</span>
          <div style={{ display: "flex", gap: 4, flexWrap: "wrap" }}>
            {uchars.slice(0, 10).map((ch) => (
              <button
                key={ch}
                className={`btn ${ch === char ? "" : "btn-secondary"}`}
                style={{ padding: "3px 8px", minWidth: 28 }}
                onClick={() => setChar(ch)}
              >
                {ch}
              </button>
            ))}
          </div>
          <span style={{ fontSize: 12, color: "var(--text-dim)", marginLeft: 12 }}>Position:</span>
          <select
            value={pos}
            onChange={(e) => setPos(Number(e.target.value))}
            style={{
              background: "var(--surface2)", border: "1px solid var(--border-hover)",
              color: "var(--text)", padding: "4px 8px", borderRadius: 4, fontFamily: "inherit",
            }}
          >
            {Array.from({ length: 8 }, (_, i) => (
              <option key={i} value={i}>pos {i}</option>
            ))}
          </select>
        </div>
      </div>

      {/* Step-by-step flow */}
      <div className="panel">
        <div className="panel-title">Data flowing through the model</div>
        <div className="explain">
          Each box shows the data at that stage. The 16 numbers get transformed at each step.
          Colors show values: <span style={{ color: "var(--red)" }}>negative</span> to{" "}
          <span style={{ color: "var(--green)" }}>positive</span>.
        </div>

        <div className="flow">
          <div className="flow-step">
            <div className="label">Token '{char}'</div>
            <div className="values">
              wte[{tokenId}]<br />
              Look up this character's<br />embedding from the table
            </div>
          </div>
          <div className="flow-arrow">→</div>
          <div className="flow-step">
            <div className="label">Position {pos}</div>
            <div className="values">
              wpe[{pos}]<br />
              Look up this position's<br />embedding from the table
            </div>
          </div>
          <div className="flow-arrow">→</div>
          <div className="flow-step">
            <div className="label">tok + pos</div>
            <div className="values">
              Add element-wise<br />
              Now encodes both<br />"what" and "where"
            </div>
          </div>
          <div className="flow-arrow">→</div>
          <div className="flow-step">
            <div className="label">RMSNorm</div>
            <div className="values">
              Normalize the vector<br />
              Keeps values in a<br />stable range
            </div>
          </div>
          <div className="flow-arrow">→</div>
          <div className="flow-step">
            <div className="label">Attention</div>
            <div className="values">
              Q = "what am I looking for?"<br />
              K = "what do I contain?"<br />
              V = "what do I offer?"<br />
              {N_HEAD} heads, each dim {16 / N_HEAD}
            </div>
          </div>
          <div className="flow-arrow">→</div>
          <div className="flow-step">
            <div className="label">MLP</div>
            <div className="values">
              Linear → ReLU → Linear<br />
              Expands to 64 dims,<br />then back to 16<br />
              <span className="highlight">{trace.mlpActiveMask?.filter(Boolean).length}/64 neurons active</span>
            </div>
          </div>
          <div className="flow-arrow">→</div>
          <div className="flow-step">
            <div className="label">Output</div>
            <div className="values">
              lm_head: 16 → 27 logits<br />
              softmax → probabilities<br />
              <span className="highlight">Top: '{top5[0]?.char}' {(top5[0]?.prob * 100).toFixed(0)}%</span>
            </div>
          </div>
        </div>
      </div>

      {/* Detailed vectors */}
      <div className="panel-row">
        <div className="panel">
          <div className="panel-title">Intermediate Vectors (16 dims)</div>
          <VectorBar values={trace.tokEmb} label={`Token embedding: wte['${char}']`} />
          <VectorBar values={trace.posEmb} label={`Position embedding: wpe[${pos}]`} />
          <VectorBar values={trace.combined} label="Combined (tok + pos)" />
          <VectorBar values={trace.afterNorm} label="After RMSNorm" />
          <VectorBar values={trace.afterAttn || []} label="After Attention + Residual" />
          <VectorBar values={trace.afterMlp || []} label="After MLP + Residual" />
        </div>

        <div className="panel">
          <div className="panel-title">Output: Next Token Probabilities</div>
          <div className="explain">
            The model's prediction for what character comes after <b>'{char}'</b> at position {pos}.
            Higher bars = more likely.
          </div>
          {top5.map((t) => (
            <div className="prob-row" key={t.id}>
              <span className="prob-label" style={t.char === "BOS" ? { color: "var(--red)", fontSize: 10 } : {}}>
                {t.char}
              </span>
              <div className="prob-bar-bg">
                <div
                  className="prob-bar"
                  style={{
                    width: `${(t.prob / maxProb) * 100}%`,
                    background: t.char === "BOS" ? "var(--red)" : "var(--blue)",
                  }}
                />
              </div>
              <span className="prob-val">{(t.prob * 100).toFixed(1)}%</span>
            </div>
          ))}
        </div>
      </div>

      {/* Attention weights */}
      {trace.attnWeights && (
        <div className="panel">
          <div className="panel-title">Attention Weights ({N_HEAD} heads)</div>
          <div className="explain">
            Each head learns to focus on different aspects. Since this is the first token,
            all heads have weight <b>1.0</b> on themselves (nothing else to attend to).
            With more tokens in the sequence, you'd see the attention distributed across previous tokens.
          </div>
          <div style={{ display: "flex", gap: 24 }}>
            {trace.attnWeights.map((hw, h) => (
              <div key={h}>
                <div style={{ fontSize: 11, color: "var(--purple)", marginBottom: 4 }}>Head {h}</div>
                <div style={{ display: "flex", gap: 2 }}>
                  {hw.map((w, t) => (
                    <div
                      key={t}
                      style={{
                        width: 28, height: 28, borderRadius: 4,
                        background: `rgba(122, 162, 247, ${w})`,
                        display: "flex", alignItems: "center", justifyContent: "center",
                        fontSize: 8, color: w > 0.3 ? "#fff" : "var(--text-dim)",
                      }}
                    >
                      {w.toFixed(2)}
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* MLP activation */}
      {trace.mlpHidden && (
        <div className="panel">
          <div className="panel-title">MLP Hidden Layer (64 neurons)</div>
          <div className="explain">
            After the linear layer expands from 16 → 64 dimensions, <b>ReLU</b> activation sets
            all negative values to zero. Only the "active" neurons (green) pass information through.
            This is how the model creates non-linear representations.
          </div>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 2 }}>
            {trace.mlpHidden.map((v, i) => (
              <div
                key={i}
                style={{
                  width: 20, height: 20, borderRadius: 3, fontSize: 6,
                  background: v > 0 ? `rgba(158, 206, 106, ${Math.min(1, v * 2)})` : "var(--surface2)",
                  display: "flex", alignItems: "center", justifyContent: "center",
                  color: v > 0 ? "#fff" : "var(--text-dim)",
                }}
                title={`neuron ${i}: ${v.toFixed(4)} ${v > 0 ? "(active)" : "(dead)"}`}
              >
                {v > 0 ? "+" : "·"}
              </div>
            ))}
          </div>
          <div style={{ fontSize: 11, color: "var(--text-dim)", marginTop: 4 }}>
            {trace.mlpActiveMask.filter(Boolean).length} / 64 neurons active after ReLU
          </div>
        </div>
      )}
    </>
  );
}
