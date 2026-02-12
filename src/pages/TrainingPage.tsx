import { useRef, useState } from "react";
import { type ModelState, trainStep, type TrainStepResult, tokenLabel } from "../engine/model";
import LossChart from "../components/LossChart";

interface Props {
  model: ModelState;
  onUpdate: () => void;
  onReset: () => void;
}

export default function TrainingPage({ model, onUpdate, onReset }: Props) {
  const [training, setTraining] = useState(false);
  const [lastResult, setLastResult] = useState<TrainStepResult | null>(null);
  const stopRef = useRef(false);

  const runTraining = (steps: number) => {
    if (training) return;
    setTraining(true);
    stopRef.current = false;

    const targetSteps = model.totalStep + steps;
    let done = 0;

    const tick = () => {
      if (stopRef.current || done >= steps) {
        setTraining(false);
        onUpdate();
        return;
      }

      const batch = Math.min(5, steps - done);
      let result: TrainStepResult | null = null;
      for (let i = 0; i < batch; i++) {
        result = trainStep(model, targetSteps);
        done++;
      }
      if (result) setLastResult(result);
      onUpdate();
      requestAnimationFrame(tick);
    };

    requestAnimationFrame(tick);
  };

  const stop = () => {
    stopRef.current = true;
  };

  return (
    <>
      <h1 className="page-title">4. Training</h1>
      <p className="page-desc">
        Each step: pick a name, feed its characters one by one, measure how wrong the predictions are
        (loss), then adjust all parameters to reduce the error. Watch the loss drop from ~3.3 (random guessing)
        as the model learns character patterns.
      </p>

      {/* Controls */}
      <div className="panel">
        <div className="panel-title">Controls</div>
        <div className="controls">
          <button className="btn" onClick={() => runTraining(200)} disabled={training}>
            Train 200 Steps
          </button>
          <button className="btn btn-secondary" onClick={() => runTraining(500)} disabled={training}>
            Train 500
          </button>
          <button className="btn btn-secondary" onClick={() => runTraining(1000)} disabled={training}>
            Train 1000
          </button>
          {training && (
            <button className="btn" style={{ background: "var(--red)" }} onClick={stop}>
              Stop
            </button>
          )}
          <button className="btn btn-secondary" onClick={onReset} disabled={training}>
            Reset Model
          </button>
        </div>
        <div className="controls">
          <span className="stat">
            Step: <b>{model.totalStep}</b>
          </span>
          <span className="stat">
            Loss: <b>{lastResult ? lastResult.loss.toFixed(4) : "—"}</b>
          </span>
          <span className="stat">
            LR: <b>{lastResult ? lastResult.lr.toFixed(6) : "0.010000"}</b>
          </span>
          <span className="stat">
            Name: <b>{lastResult?.doc ?? "—"}</b>
          </span>
        </div>
      </div>

      {/* Loss chart */}
      <div className="panel">
        <div className="panel-title">Loss Over Time</div>
        <div className="explain">
          <b>Loss</b> measures how wrong the model's predictions are. Lower = better.<br />
          <span style={{ color: "var(--red)" }}>Red dashed line</span> = random guessing (~3.30 for 27 tokens).<br />
          <span style={{ color: "var(--green)" }}>Green line</span> = moving average (smoothed trend).
        </div>
        <LossChart lossHistory={model.lossHistory} />
      </div>

      {/* Last training step details */}
      {lastResult && (
        <div className="panel">
          <div className="panel-title">Last Training Step Detail</div>
          <div className="explain">
            The model saw the name <b>"{lastResult.doc}"</b> and tried to predict each next character.
            Below is the loss at each position — higher loss means the model was more surprised.
          </div>

          {/* Token sequence */}
          <div className="token-flow" style={{ marginTop: 0, marginBottom: 12 }}>
            {lastResult.tokens.map((t, i) => (
              <span key={i} style={{ display: "contents" }}>
                {i > 0 && <span className="arrow-sym">→</span>}
                <div className={`token-box ${t === 26 ? "bos" : ""}`} style={{ padding: "4px 6px" }}>
                  <span className="char" style={{ fontSize: 14 }}>{tokenLabel(t)}</span>
                  <span className="id">{t}</span>
                </div>
              </span>
            ))}
          </div>

          {/* Per-position loss */}
          <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
            {lastResult.perPositionLoss.map((loss, i) => {
              const from = tokenLabel(lastResult.tokens[i]);
              const to = tokenLabel(lastResult.tokens[i + 1]);
              const intensity = Math.min(1, loss / 4);
              return (
                <div
                  key={i}
                  style={{
                    padding: "6px 10px",
                    background: `rgba(247, 118, 142, ${intensity * 0.3})`,
                    border: `1px solid rgba(247, 118, 142, ${intensity * 0.5})`,
                    borderRadius: 6,
                    fontSize: 11,
                    textAlign: "center",
                  }}
                >
                  <div>
                    <span style={{ color: "var(--cyan)" }}>{from}</span>
                    <span style={{ color: "var(--text-dim)" }}> → </span>
                    <span style={{ color: "var(--green)" }}>{to}</span>
                  </div>
                  <div style={{ color: "var(--red)", fontWeight: "bold", fontSize: 10 }}>
                    loss: {loss.toFixed(3)}
                  </div>
                </div>
              );
            })}
          </div>
          <div style={{ fontSize: 11, color: "var(--text-dim)", marginTop: 8 }}>
            Average loss: {lastResult.loss.toFixed(4)} | The redder the box, the more surprised the model was
          </div>
        </div>
      )}

      {/* What's happening */}
      <div className="panel">
        <div className="panel-title">What happens in each training step</div>
        <div className="explain">
          <b>1. Forward pass:</b> Feed each token through the model, get predictions for the next token.<br />
          <b>2. Compute loss:</b> Measure how wrong each prediction was using <code>-log(P(correct))</code>.<br />
          <b>3. Backward pass:</b> Calculate gradients — which direction should each of the {model.params.length} parameters
          move to reduce the loss?<br />
          <b>4. Adam update:</b> Adjust each parameter by a small amount in the right direction. The learning rate
          starts at 0.01 and linearly decays to 0.<br /><br />
          After enough steps, the model learns patterns like: after 'e', 'm' is likely; names often end with 'a', 'n', 'y'; etc.
        </div>
      </div>
    </>
  );
}
