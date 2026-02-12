import { useState } from "react";
import { type ModelState, uchars, N_EMBD, BLOCK_SIZE, charToId } from "../engine/model";
import Heatmap, { VectorBar } from "../components/Heatmap";

interface Props {
  model: ModelState;
}

export default function EmbeddingsPage({ model }: Props) {
  const [hoverRow, setHoverRow] = useState<number | null>(null);
  const [selectedChar, setSelectedChar] = useState("e");

  const wte = model.stateDict.wte;
  const wpe = model.stateDict.wpe;
  const wteLabels = [...uchars, "BOS"];
  const wpeLabels = Array.from({ length: BLOCK_SIZE }, (_, i) => `p${i}`);

  const charId = charToId[selectedChar] ?? 0;
  const tokEmb = wte[charId].map((v) => v.data);
  const posEmb = wpe[0].map((v) => v.data); // position 0
  const combined = tokEmb.map((t, i) => t + posEmb[i]);

  return (
    <>
      <h1 className="page-title">2. Embeddings (wte & wpe)</h1>
      <p className="page-desc">
        The model represents each token and each position as a list of 16 numbers (a "vector").
        These numbers are the model's learnable parameters — they start random and get adjusted during training.
      </p>

      {/* WTE */}
      <div className="panel">
        <div className="panel-title">wte — Token Embeddings</div>
        <div className="explain">
          <b>wte</b> stands for "Word Token Embeddings". It's a table with <b>{wte.length} rows</b> (one per token)
          and <b>{N_EMBD} columns</b> (the embedding dimension).<br /><br />
          Each row is how the model "sees" that character. Think of it as the model's internal representation
          of each letter. <b>Hover over a row</b> to highlight it.<br /><br />
          Colors: <span style={{ color: "var(--red)" }}>red = negative</span>,{" "}
          <span style={{ color: "var(--text-dim)" }}>dark = near zero</span>,{" "}
          <span style={{ color: "var(--green)" }}>green = positive</span>.
          Right now these are <b>random</b> — after training, similar letters will have similar patterns.
        </div>
        <Heatmap
          matrix={wte}
          rowLabels={wteLabels}
          colCount={N_EMBD}
          highlightRow={hoverRow ?? undefined}
          onHoverRow={setHoverRow}
        />
      </div>

      {/* WPE */}
      <div className="panel">
        <div className="panel-title">wpe — Position Embeddings</div>
        <div className="explain">
          <b>wpe</b> stands for "Word Position Embeddings". It tells the model <b>where</b> a token is
          in the sequence. Position 0 = first character, position 1 = second, etc.<br /><br />
          Without this, the model couldn't distinguish "ab" from "ba" — both have the same characters!
          The position embedding is <b>added</b> to the token embedding.
        </div>
        <Heatmap matrix={wpe} rowLabels={wpeLabels} colCount={N_EMBD} />
      </div>

      {/* Interactive: how they combine */}
      <div className="panel">
        <div className="panel-title">How wte + wpe combine</div>
        <div className="explain">
          When the model processes a token, it looks up <code>wte[token_id]</code> and <code>wpe[position]</code>,
          then <b>adds them together</b> element-by-element. The result is a single vector that encodes
          both <b>which</b> character and <b>where</b> it is.
        </div>

        <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 12 }}>
          {uchars.map((ch) => (
            <button
              key={ch}
              className={`btn ${ch === selectedChar ? "" : "btn-secondary"}`}
              style={{ padding: "4px 10px", fontSize: 14, minWidth: 32 }}
              onClick={() => setSelectedChar(ch)}
            >
              {ch}
            </button>
          ))}
        </div>

        <div style={{ fontSize: 13, color: "var(--purple)", marginBottom: 8 }}>
          '{selectedChar}' at position 0:
        </div>

        <VectorBar values={tokEmb} label={`wte['${selectedChar}'] (token embedding)`} />
        <div style={{ fontSize: 14, color: "var(--text-dim)", textAlign: "center", margin: "4px 0" }}>+</div>
        <VectorBar values={posEmb} label="wpe[0] (position embedding)" />
        <div style={{ fontSize: 14, color: "var(--text-dim)", textAlign: "center", margin: "4px 0" }}>=</div>
        <VectorBar values={combined} label="combined (input to the model)" />
      </div>
    </>
  );
}
