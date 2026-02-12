import { useState } from "react";
import { type ModelState, generateName, type InferenceStep, tokenLabel } from "../engine/model";

interface GeneratedResult {
  name: string;
  steps: InferenceStep[];
}

export default function InferencePage({ model }: { model: ModelState }) {
  const [temperature, setTemperature] = useState(5); // x10
  const [results, setResults] = useState<GeneratedResult[]>([]);
  const [activeTrace, setActiveTrace] = useState<GeneratedResult | null>(null);
  const [activeStep, setActiveStep] = useState(0);

  const temp = temperature / 10;

  const generate = (count: number) => {
    const newResults: GeneratedResult[] = [];
    for (let i = 0; i < count; i++) {
      newResults.push(generateName(model, temp));
    }
    setResults((prev) => [...newResults, ...prev]);
    if (newResults[0]) {
      setActiveTrace(newResults[0]);
      setActiveStep(0);
    }
  };

  const step = activeTrace?.steps[activeStep];
  const top10 = step
    ? step.probs
        .map((p, i) => ({ id: i, char: tokenLabel(i), prob: p }))
        .sort((a, b) => b.prob - a.prob)
        .slice(0, 12)
    : [];
  const maxProb = Math.max(...top10.map((t) => t.prob), 0.01);

  return (
    <>
      <h1 className="page-title">5. Inference</h1>
      <p className="page-desc">
        After training, the model generates new names it has never seen. Starting from BOS, it
        predicts the next character, samples one, feeds it back in, and repeats until BOS appears again.
      </p>

      {/* Controls */}
      <div className="panel">
        <div className="panel-title">Generate</div>
        {model.totalStep === 0 && (
          <div className="explain" style={{ borderLeftColor: "var(--orange)" }}>
            The model hasn't been trained yet! Go to the <b>Training</b> tab first and train for at least 200 steps.
            You can still generate, but results will be random gibberish.
          </div>
        )}
        <div className="controls">
          <button className="btn" onClick={() => generate(1)}>
            Generate 1
          </button>
          <button className="btn btn-secondary" onClick={() => generate(10)}>
            Generate 10
          </button>
          <button className="btn btn-secondary" onClick={() => setResults([])}>
            Clear
          </button>
          <span style={{ fontSize: 12, color: "var(--text-dim)", marginLeft: 8 }}>
            Temperature:
          </span>
          <input
            type="range"
            min="1"
            max="20"
            value={temperature}
            onChange={(e) => setTemperature(Number(e.target.value))}
            style={{ width: 100 }}
          />
          <span className="stat">
            <b>{temp.toFixed(1)}</b>
          </span>
        </div>
        <div className="explain">
          <b>Temperature</b> controls randomness. Low (0.1) = always picks the most likely character (boring but safe).
          High (2.0) = more random picks (creative but chaotic). Try different values!
        </div>
      </div>

      {/* Generated names */}
      <div className="panel">
        <div className="panel-title">Generated Names ({results.length})</div>
        <div className="gen-names">
          {results.map((r, i) => (
            <span
              key={i}
              className="gen-name"
              style={{
                cursor: "pointer",
                borderColor: activeTrace === r ? "var(--blue)" : undefined,
              }}
              onClick={() => {
                setActiveTrace(r);
                setActiveStep(0);
              }}
            >
              {r.name || "(empty)"}
            </span>
          ))}
          {results.length === 0 && (
            <span style={{ fontSize: 12, color: "var(--text-dim)" }}>
              Click "Generate" to create names...
            </span>
          )}
        </div>
      </div>

      {/* Step-by-step trace */}
      {activeTrace && (
        <div className="panel-row">
          <div className="panel">
            <div className="panel-title">
              Generation Trace: "{activeTrace.name}"
            </div>
            <div className="explain">
              Click each step to see what the model was "thinking" at that position.
              The model picks from the probability distribution at each step.
            </div>

            <div style={{ display: "flex", gap: 4, flexWrap: "wrap", marginBottom: 12 }}>
              {activeTrace.steps.map((s, i) => (
                <button
                  key={i}
                  className={`btn ${i === activeStep ? "" : "btn-secondary"}`}
                  style={{ padding: "4px 10px", fontSize: 12 }}
                  onClick={() => setActiveStep(i)}
                >
                  pos {s.pos}:{" "}
                  <span style={{ fontWeight: "bold" }}>{s.chosenChar}</span>
                </button>
              ))}
            </div>

            {step && (
              <div className="trace">
                {activeTrace.steps.map((s, i) => (
                  <div key={i} style={{ opacity: i === activeStep ? 1 : 0.5 }}>
                    <span style={{ color: "var(--text-dim)" }}>pos {s.pos}: </span>
                    <span className="candidates">
                      [{s.top5.map((t) => `${t.char}:${(t.prob * 100).toFixed(0)}%`).join(", ")}]
                    </span>
                    <span style={{ color: "var(--text-dim)" }}> → </span>
                    <span className="picked">'{s.chosenChar}'</span>
                    {s.chosenChar === "BOS" && (
                      <span style={{ color: "var(--red)" }}> END</span>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>

          <div className="panel">
            <div className="panel-title">
              Probabilities at position {step?.pos ?? 0}
            </div>
            <div className="explain">
              The model outputs these probabilities for the next character.
              The <span style={{ color: "var(--green)" }}>highlighted</span> one was sampled.
            </div>
            {top10.map((t) => (
              <div className="prob-row" key={t.id}>
                <span
                  className="prob-label"
                  style={{
                    color:
                      t.id === step?.chosenId
                        ? "var(--green)"
                        : t.char === "BOS"
                        ? "var(--red)"
                        : "var(--cyan)",
                    fontSize: t.char === "BOS" ? 9 : 13,
                  }}
                >
                  {t.char}
                </span>
                <div className="prob-bar-bg">
                  <div
                    className="prob-bar"
                    style={{
                      width: `${(t.prob / maxProb) * 100}%`,
                      background:
                        t.id === step?.chosenId
                          ? "var(--green)"
                          : t.char === "BOS"
                          ? "var(--red)"
                          : "var(--blue)",
                    }}
                  />
                </div>
                <span className="prob-val">
                  {(t.prob * 100).toFixed(1)}%
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* How inference works */}
      <div className="panel">
        <div className="panel-title">How generation works</div>
        <div className="explain">
          <b>1.</b> Start with the BOS token (signals "beginning of a name").<br />
          <b>2.</b> Feed it through the model → get probabilities for all 27 possible next tokens.<br />
          <b>3.</b> <b>Sample</b> one token from this distribution (temperature affects the randomness).<br />
          <b>4.</b> If the sampled token is BOS → stop (end of name).<br />
          <b>5.</b> Otherwise, feed the sampled token back as input and repeat from step 2.<br /><br />
          This is called <b>autoregressive generation</b> — the model generates one token at a time,
          each depending on all previous tokens. This is exactly how ChatGPT works, just at a much larger scale.
        </div>
      </div>
    </>
  );
}
