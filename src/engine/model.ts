/**
 * MicroGPT model — a complete tiny GPT in TypeScript.
 * Mirrors microgpt.py exactly: same architecture, same math.
 */
import { Value, vAdd, vSum } from "./autograd";
import { createRng, gaussRandom, shuffle } from "./random";
import { NAMES_RAW } from "./data";

// ─── Config ───
export const N_EMBD = 16;
export const N_HEAD = 4;
export const N_LAYER = 1;
export const BLOCK_SIZE = 16;
export const HEAD_DIM = N_EMBD / N_HEAD;

// ─── Tokenizer ───
const allDocs = NAMES_RAW.split(",")
  .map((s) => s.trim())
  .filter(Boolean);

export const uchars = [...new Set(allDocs.join(""))].sort();
export const BOS = uchars.length;
export const vocabSize = uchars.length + 1;
export const charToId: Record<string, number> = Object.fromEntries(
  uchars.map((c, i) => [c, i])
);

export function tokenize(name: string): number[] {
  return [BOS, ...name.split("").map((c) => charToId[c] ?? 0), BOS];
}

export function tokenLabel(id: number): string {
  return id < uchars.length ? uchars[id] : "BOS";
}

// ─── State ───
export interface ModelState {
  stateDict: Record<string, Value[][]>;
  params: Value[];
  adamM: Float64Array;
  adamV: Float64Array;
  totalStep: number;
  lossHistory: number[];
  docs: string[];
  rng: () => number;
}

function makeMatrix(
  nout: number,
  nin: number,
  rng: () => number,
  std = 0.08
): Value[][] {
  return Array.from({ length: nout }, () =>
    Array.from({ length: nin }, () => new Value(gaussRandom(rng) * std))
  );
}

export function createModel(): ModelState {
  const rng = createRng(42);
  const docs = [...allDocs];
  shuffle(docs, rng);

  const stateDict: Record<string, Value[][]> = {
    wte: makeMatrix(vocabSize, N_EMBD, rng),
    wpe: makeMatrix(BLOCK_SIZE, N_EMBD, rng),
    lm_head: makeMatrix(vocabSize, N_EMBD, rng),
  };

  for (let i = 0; i < N_LAYER; i++) {
    stateDict[`layer${i}.attn_wq`] = makeMatrix(N_EMBD, N_EMBD, rng);
    stateDict[`layer${i}.attn_wk`] = makeMatrix(N_EMBD, N_EMBD, rng);
    stateDict[`layer${i}.attn_wv`] = makeMatrix(N_EMBD, N_EMBD, rng);
    stateDict[`layer${i}.attn_wo`] = makeMatrix(N_EMBD, N_EMBD, rng);
    stateDict[`layer${i}.mlp_fc1`] = makeMatrix(4 * N_EMBD, N_EMBD, rng);
    stateDict[`layer${i}.mlp_fc2`] = makeMatrix(N_EMBD, 4 * N_EMBD, rng);
  }

  const params: Value[] = [];
  for (const key of Object.keys(stateDict)) {
    for (const row of stateDict[key]) {
      for (const p of row) params.push(p);
    }
  }

  return {
    stateDict,
    params,
    adamM: new Float64Array(params.length),
    adamV: new Float64Array(params.length),
    totalStep: 0,
    lossHistory: [],
    docs,
    rng,
  };
}

// ─── Layers ───
export function linear(x: Value[], w: Value[][]): Value[] {
  return w.map((wo) => vSum(wo.map((wi, i) => wi.mul(x[i]))));
}

export function softmax(logits: Value[]): Value[] {
  const maxVal = Math.max(...logits.map((v) => v.data));
  const exps = logits.map((v) => v.sub(maxVal).exp());
  const total = vSum(exps);
  return exps.map((e) => e.div(total));
}

export function rmsnorm(x: Value[]): Value[] {
  const ms = vSum(x.map((xi) => xi.mul(xi))).div(x.length);
  const scale = ms.add(1e-5).pow(-0.5);
  return x.map((xi) => xi.mul(scale));
}

// ─── Forward pass (returns logits + intermediate states for visualization) ───
export interface ForwardTrace {
  tokenId: number;
  posId: number;
  tokEmb: number[];
  posEmb: number[];
  combined: number[];
  afterNorm: number[];
  q: number[];
  k: number[];
  v: number[];
  attnWeights: number[][]; // [head][time]
  afterAttn: number[];
  mlpHidden: number[];
  mlpActiveMask: boolean[];
  afterMlp: number[];
  logits: number[];
  probs: number[];
}

export function gptForward(
  tokenId: number,
  posId: number,
  keys: Value[][][],
  values: Value[][][],
  state: ModelState,
  trace = false
): { logits: Value[]; trace?: ForwardTrace } {
  const { stateDict } = state;
  const tokEmb = stateDict.wte[tokenId];
  const posEmb = stateDict.wpe[posId];
  let x = vAdd(tokEmb, posEmb);

  const t: Partial<ForwardTrace> = trace
    ? {
        tokenId,
        posId,
        tokEmb: tokEmb.map((v) => v.data),
        posEmb: posEmb.map((v) => v.data),
        combined: x.map((v) => v.data),
      }
    : {};

  x = rmsnorm(x);
  if (trace) t.afterNorm = x.map((v) => v.data);

  for (let li = 0; li < N_LAYER; li++) {
    const xRes = x;
    x = rmsnorm(x);
    const q = linear(x, stateDict[`layer${li}.attn_wq`]);
    const k = linear(x, stateDict[`layer${li}.attn_wk`]);
    const v = linear(x, stateDict[`layer${li}.attn_wv`]);
    keys[li].push(k);
    values[li].push(v);

    if (trace) {
      t.q = q.map((v) => v.data);
      t.k = k.map((v) => v.data);
      t.v = v.map((v) => v.data);
    }

    const xAttn: Value[] = [];
    const allAttnWeights: number[][] = [];

    for (let h = 0; h < N_HEAD; h++) {
      const hs = h * HEAD_DIM;
      const qH = q.slice(hs, hs + HEAD_DIM);
      const kH = keys[li].map((ki) => ki.slice(hs, hs + HEAD_DIM));
      const vH = values[li].map((vi) => vi.slice(hs, hs + HEAD_DIM));
      const attnLogits = kH.map((kht) =>
        vSum(qH.map((qj, j) => qj.mul(kht[j]))).div(Math.sqrt(HEAD_DIM))
      );
      const attnW = softmax(attnLogits);

      if (trace) allAttnWeights.push(attnW.map((w) => w.data));

      for (let j = 0; j < HEAD_DIM; j++) {
        xAttn.push(vSum(attnW.map((w, ti) => w.mul(vH[ti][j]))));
      }
    }

    if (trace) t.attnWeights = allAttnWeights;

    x = linear(xAttn, stateDict[`layer${li}.attn_wo`]);
    x = vAdd(x, xRes);
    if (trace) t.afterAttn = x.map((v) => v.data);

    const xRes2 = x;
    x = rmsnorm(x);
    x = linear(x, stateDict[`layer${li}.mlp_fc1`]);

    if (trace) {
      t.mlpHidden = x.map((v) => v.data);
      t.mlpActiveMask = x.map((v) => v.data > 0);
    }

    x = x.map((xi) => xi.relu());
    x = linear(x, stateDict[`layer${li}.mlp_fc2`]);
    x = vAdd(x, xRes2);
    if (trace) t.afterMlp = x.map((v) => v.data);
  }

  const logits = linear(x, stateDict.lm_head);
  if (trace) {
    t.logits = logits.map((v) => v.data);
    const probs = softmax(logits);
    t.probs = probs.map((p) => p.data);
  }

  return { logits, trace: trace ? (t as ForwardTrace) : undefined };
}

// ─── Train one step ───
export interface TrainStepResult {
  loss: number;
  doc: string;
  lr: number;
  tokens: number[];
  perPositionLoss: number[];
}

const LR0 = 0.01,
  BETA1 = 0.85,
  BETA2 = 0.99,
  EPS = 1e-8;

export function trainStep(
  state: ModelState,
  totalTargetSteps: number
): TrainStepResult {
  const doc = state.docs[state.totalStep % state.docs.length];
  const tokens = tokenize(doc);
  const n = Math.min(BLOCK_SIZE, tokens.length - 1);

  const keys: Value[][][] = Array.from({ length: N_LAYER }, () => []);
  const vals: Value[][][] = Array.from({ length: N_LAYER }, () => []);
  const losses: Value[] = [];
  const perPositionLoss: number[] = [];

  for (let pos = 0; pos < n; pos++) {
    const { logits } = gptForward(tokens[pos], pos, keys, vals, state);
    const probs = softmax(logits);
    const lossT = probs[tokens[pos + 1]].log().neg();
    losses.push(lossT);
    perPositionLoss.push(lossT.data);
  }

  const loss = vSum(losses).div(n);
  loss.backward();

  const lrT = LR0 * (1 - state.totalStep / totalTargetSteps);
  for (let i = 0; i < state.params.length; i++) {
    state.adamM[i] = BETA1 * state.adamM[i] + (1 - BETA1) * state.params[i].grad;
    state.adamV[i] = BETA2 * state.adamV[i] + (1 - BETA2) * state.params[i].grad ** 2;
    const mHat = state.adamM[i] / (1 - BETA1 ** (state.totalStep + 1));
    const vHat = state.adamV[i] / (1 - BETA2 ** (state.totalStep + 1));
    state.params[i].data -= lrT * mHat / (Math.sqrt(vHat) + EPS);
    state.params[i].grad = 0;
  }

  state.totalStep++;
  state.lossHistory.push(loss.data);

  return { loss: loss.data, doc, lr: lrT, tokens, perPositionLoss };
}

// ─── Inference ───
export interface InferenceStep {
  pos: number;
  probs: number[];
  chosenId: number;
  chosenChar: string;
  top5: { id: number; char: string; prob: number }[];
}

export function generateName(
  state: ModelState,
  temperature = 0.5
): { name: string; steps: InferenceStep[] } {
  const keys: Value[][][] = Array.from({ length: N_LAYER }, () => []);
  const vals: Value[][][] = Array.from({ length: N_LAYER }, () => []);
  let tokenId = BOS;
  const sample: string[] = [];
  const steps: InferenceStep[] = [];

  for (let pos = 0; pos < BLOCK_SIZE; pos++) {
    const { logits } = gptForward(tokenId, pos, keys, vals, state);
    const scaledLogits = logits.map((l) => new Value(l.data / temperature));
    const probs = softmax(scaledLogits);
    const probData = probs.map((p) => p.data);

    // Sample
    const r = state.rng();
    let cum = 0;
    let chosen = vocabSize - 1;
    for (let i = 0; i < vocabSize; i++) {
      cum += probData[i];
      if (r < cum) {
        chosen = i;
        break;
      }
    }

    const top5 = probData
      .map((p, i) => ({ id: i, char: tokenLabel(i), prob: p }))
      .sort((a, b) => b.prob - a.prob)
      .slice(0, 5);

    steps.push({
      pos,
      probs: probData,
      chosenId: chosen,
      chosenChar: tokenLabel(chosen),
      top5,
    });

    if (chosen === BOS) break;
    sample.push(uchars[chosen]);
    tokenId = chosen;
  }

  return { name: sample.join(""), steps };
}
