import { useEffect, useRef } from "react";

interface Props {
  lossHistory: number[];
}

function getCssVar(name: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

export default function LossChart({ lossHistory }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d")!;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = canvas.clientWidth * dpr;
    canvas.height = canvas.clientHeight * dpr;
    ctx.scale(dpr, dpr);
    const W = canvas.clientWidth;
    const H = canvas.clientHeight;

    // Read theme colors from CSS variables
    const colSurface2 = getCssVar("--surface2");
    const colBorder = getCssVar("--border");
    const colTextDim = getCssVar("--text-dim");
    const colBlue = getCssVar("--blue");
    const colGreen = getCssVar("--green");
    const colRed = getCssVar("--red");

    ctx.fillStyle = colSurface2;
    ctx.fillRect(0, 0, W, H);

    if (lossHistory.length === 0) {
      ctx.fillStyle = colTextDim;
      ctx.font = "14px monospace";
      ctx.textAlign = "center";
      ctx.fillText('Press "Train" to start', W / 2, H / 2);
      return;
    }

    const pad = { top: 20, right: 20, bottom: 30, left: 50 };
    const pw = W - pad.left - pad.right;
    const ph = H - pad.top - pad.bottom;

    const minLoss = Math.min(...lossHistory) * 0.95;
    const maxLoss = Math.max(...lossHistory) * 1.02;

    // Grid
    ctx.strokeStyle = colBorder;
    ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
      const y = pad.top + (ph * i) / 4;
      ctx.beginPath();
      ctx.moveTo(pad.left, y);
      ctx.lineTo(W - pad.right, y);
      ctx.stroke();
      ctx.fillStyle = colTextDim;
      ctx.font = "10px monospace";
      ctx.textAlign = "right";
      const val = maxLoss - ((maxLoss - minLoss) * i) / 4;
      ctx.fillText(val.toFixed(2), pad.left - 6, y + 4);
    }

    // Raw loss line
    ctx.beginPath();
    ctx.strokeStyle = colBlue + "44";
    ctx.lineWidth = 1;
    for (let i = 0; i < lossHistory.length; i++) {
      const x = pad.left + (i / Math.max(1, lossHistory.length - 1)) * pw;
      const y = pad.top + (1 - (lossHistory[i] - minLoss) / (maxLoss - minLoss)) * ph;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Moving average
    if (lossHistory.length > 10) {
      ctx.beginPath();
      ctx.strokeStyle = colGreen;
      ctx.lineWidth = 2;
      const ws = Math.min(20, Math.floor(lossHistory.length / 3));
      for (let i = ws; i < lossHistory.length; i++) {
        const avg = lossHistory.slice(i - ws, i).reduce((a, b) => a + b, 0) / ws;
        const x = pad.left + (i / (lossHistory.length - 1)) * pw;
        const y = pad.top + (1 - (avg - minLoss) / (maxLoss - minLoss)) * ph;
        if (i === ws) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
    }

    // Random baseline
    const vocabSize = 27;
    const randomLoss = Math.log(vocabSize);
    if (randomLoss >= minLoss && randomLoss <= maxLoss) {
      const y = pad.top + (1 - (randomLoss - minLoss) / (maxLoss - minLoss)) * ph;
      ctx.strokeStyle = colRed + "44";
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(pad.left, y);
      ctx.lineTo(W - pad.right, y);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = colRed;
      ctx.textAlign = "left";
      ctx.font = "10px monospace";
      ctx.fillText("random baseline (3.30)", pad.left + 4, y - 4);
    }

    // Labels
    ctx.fillStyle = colTextDim;
    ctx.font = "10px monospace";
    ctx.textAlign = "center";
    ctx.fillText(`Step ${lossHistory.length}`, W / 2, H - 6);
    ctx.textAlign = "right";
    ctx.fillStyle = colBlue;
    ctx.fillText("raw", W - pad.right, pad.top - 6);
    ctx.fillStyle = colGreen;
    ctx.fillText("avg", W - pad.right - 40, pad.top - 6);
  }, [lossHistory, lossHistory.length]);

  return <canvas ref={canvasRef} className="chart" />;
}
