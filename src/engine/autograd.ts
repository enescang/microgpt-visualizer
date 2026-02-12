/**
 * Autograd engine — port of microgpt.py's Value class.
 * Each Value wraps a scalar and tracks how it was computed,
 * enabling automatic gradient calculation via backpropagation.
 */
export class Value {
  data: number;
  grad: number;
  _children: Value[];
  _localGrads: number[];
  label: string;

  constructor(
    data: number,
    children: Value[] = [],
    localGrads: number[] = [],
    label = ""
  ) {
    this.data = data;
    this.grad = 0;
    this._children = children;
    this._localGrads = localGrads;
    this.label = label;
  }

  add(other: Value | number): Value {
    const o = other instanceof Value ? other : new Value(other);
    return new Value(this.data + o.data, [this, o], [1, 1]);
  }

  mul(other: Value | number): Value {
    const o = other instanceof Value ? other : new Value(other);
    return new Value(this.data * o.data, [this, o], [o.data, this.data]);
  }

  pow(n: number): Value {
    return new Value(this.data ** n, [this], [n * this.data ** (n - 1)]);
  }

  log(): Value {
    return new Value(Math.log(this.data), [this], [1 / this.data]);
  }

  exp(): Value {
    const e = Math.exp(this.data);
    return new Value(e, [this], [e]);
  }

  relu(): Value {
    return new Value(
      Math.max(0, this.data),
      [this],
      [this.data > 0 ? 1 : 0]
    );
  }

  neg(): Value {
    return this.mul(-1);
  }
  sub(other: Value | number): Value {
    const o = other instanceof Value ? other : new Value(other);
    return this.add(o.neg());
  }
  div(other: Value | number): Value {
    const o = other instanceof Value ? other : new Value(other);
    return this.mul(o.pow(-1));
  }

  backward(): void {
    const topo: Value[] = [];
    const visited = new Set<Value>();
    const build = (v: Value) => {
      if (visited.has(v)) return;
      visited.add(v);
      for (const child of v._children) build(child);
      topo.push(v);
    };
    build(this);
    this.grad = 1;
    for (let i = topo.length - 1; i >= 0; i--) {
      const v = topo[i];
      for (let j = 0; j < v._children.length; j++) {
        v._children[j].grad += v._localGrads[j] * v.grad;
      }
    }
  }
}

// Helpers for Value arrays
export function vAdd(a: Value[], b: Value[]): Value[] {
  return a.map((v, i) => v.add(b[i]));
}

export function vSum(arr: Value[]): Value {
  let s = arr[0];
  for (let i = 1; i < arr.length; i++) s = s.add(arr[i]);
  return s;
}
