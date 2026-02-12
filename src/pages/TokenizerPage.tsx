import { useState } from "react";
import { uchars, BOS, tokenize, tokenLabel, vocabSize } from "../engine/model";

export default function TokenizerPage() {
  const [input, setInput] = useState("emma");

  const clean = input.toLowerCase().replace(/[^a-z]/g, "").slice(0, 16);
  const tokens = clean ? tokenize(clean) : [];

  return (
    <>
      <h1 className="page-title">1. Tokenizer</h1>
      <p className="page-desc">
        The model cannot read letters. It only understands numbers. The tokenizer
        converts each character to a numeric ID, and adds a special BOS (Begin/End of Sequence)
        token to mark the boundaries of each name.
      </p>

      {/* Character → ID mapping */}
      <div className="panel">
        <div className="panel-title">Character → ID Mapping</div>
        <div className="explain">
          Every unique character in the dataset gets a <b>numeric ID</b>.
          There are <b>{uchars.length}</b> characters (a-z) plus one special <b>BOS</b> token (id={BOS}).
          Total vocabulary size: <b>{vocabSize}</b>.
        </div>
        <div style={{ display: "flex", flexWrap: "nowrap", gap: 4, paddingTop: 6, overflowX: "auto" }}>
          {uchars.map((ch, i) => (
            <div
              key={ch}
              className="token-box"
              style={{ minWidth: 34, cursor: "default" }}
            >
              <span className="char">{ch}</span>
              <span className="id">{i}</span>
            </div>
          ))}
          <div className="token-box bos" style={{ minWidth: 34, cursor: "default" }}>
            <span className="char">BOS</span>
            <span className="id">{BOS}</span>
          </div>
        </div>
      </div>

      {/* Interactive tokenizer */}
      <div className="panel">
        <div className="panel-title">Try It: Type a Name</div>
        <div className="explain">
          Type any name below. Watch how it becomes a sequence of token IDs.
          The model will learn from sequences like this — predicting each next token from the previous ones.
        </div>
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Type a name..."
          maxLength={16}
        />

        {clean && (
          <>
            <div className="token-flow">
              {tokens.map((t, i) => {
                const label = tokenLabel(t);
                const isBos = t === BOS;
                return (
                  <span key={i} style={{ display: "contents" }}>
                    {i > 0 && <span className="arrow-sym">→</span>}
                    <div className={`token-box ${isBos ? "bos" : ""}`}>
                      <span className="char">{label}</span>
                      <span className="id">id: {t}</span>
                    </div>
                  </span>
                );
              })}
            </div>

            {/* What the model must learn */}
            <div style={{ marginTop: 16 }}>
              <div className="panel-title">What the model must learn:</div>
              <div className="explain">
                For each position, the model sees the <b>current token</b> and must predict
                the <b>next token</b>. This is called <b>next-token prediction</b> — the core
                of how GPT works.
              </div>
              <div style={{ display: "flex", flexWrap: "wrap", gap: 8, marginTop: 8 }}>
                {tokens.slice(0, -1).map((t, i) => {
                  const from = tokenLabel(t);
                  const to = tokenLabel(tokens[i + 1]);
                  return (
                    <div
                      key={i}
                      style={{
                        padding: "4px 10px",
                        background: "var(--surface2)",
                        borderRadius: 6,
                        fontSize: 12,
                        border: "1px solid var(--border-hover)",
                      }}
                    >
                      <span style={{ color: "var(--cyan)" }}>{from}</span>
                      <span style={{ color: "var(--text-dim)", margin: "0 4px" }}>→</span>
                      <span style={{ color: "var(--green)" }}>{to}</span>
                    </div>
                  );
                })}
              </div>
            </div>
          </>
        )}
      </div>

      {/* Vocab details */}
      <div className="panel">
        <div className="panel-title">How tokenization works in real GPTs</div>
        <div className="explain">
          This micro GPT uses <b>character-level</b> tokenization — each letter is one token.<br /><br />
          Real GPTs (like GPT-4) use <b>subword tokenization</b> (BPE) where common word parts
          like "ing", "tion", "the" become single tokens. This makes the vocabulary much larger (~50K-100K tokens)
          but lets the model process text much more efficiently.<br /><br />
          Our tiny vocab of <b>{vocabSize}</b> tokens is enough to learn patterns in names!
        </div>
      </div>
    </>
  );
}
