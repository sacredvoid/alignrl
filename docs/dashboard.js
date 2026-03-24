/* ── alignrl results dashboard ──────────────────────────────── */

const STAGES = ["base", "sft", "grpo", "dpo"];
const STAGE_COLORS = {
  base: "#8b949e",
  sft: "#58a6ff",
  grpo: "#3fb950",
  dpo: "#d2a8ff",
};

/* ── Benchmark table ────────────────────────────────────────── */

async function loadBenchmarks() {
  const tbody = document.getElementById("results-body");
  if (!tbody) return;

  let data;
  try {
    const resp = await fetch("../results/comparison.json");
    data = await resp.json();
  } catch {
    tbody.innerHTML =
      '<tr><td colspan="5" style="text-align:center;color:var(--text-muted)">Could not load results/comparison.json</td></tr>';
    return;
  }

  for (const [benchmark, stages] of Object.entries(data)) {
    const metric = Object.keys(Object.values(stages)[0])[0];
    const scores = STAGES.map((s) => {
      const val = stages[s]?.[metric];
      return val != null ? val : null;
    });

    const best = Math.max(...scores.filter((s) => s != null));

    const row = document.createElement("tr");

    // Benchmark name
    const nameCell = document.createElement("td");
    nameCell.textContent = benchmark;
    row.appendChild(nameCell);

    // Score cells
    scores.forEach((score, i) => {
      const td = document.createElement("td");
      if (score != null) {
        const pct = (score * 100).toFixed(1);
        td.textContent = pct + "%";
        if (score === best) {
          td.classList.add("score-best");
        }
        // Add a small bar underneath
        const bar = document.createElement("div");
        bar.className = "score-bar";
        bar.style.width = pct + "%";
        bar.style.maxWidth = "100%";
        bar.style.background = STAGE_COLORS[STAGES[i]];
        td.appendChild(bar);
      } else {
        td.textContent = "-";
        td.style.color = "var(--text-muted)";
      }
      row.appendChild(td);
    });

    tbody.appendChild(row);
  }
}

/* ── Training curves (placeholder data) ─────────────────────── */

function renderCharts() {
  const gridColor = "rgba(48, 54, 61, 0.6)";
  const textColor = "#8b949e";

  const commonOptions = {
    responsive: true,
    animation: { duration: 800 },
    plugins: {
      legend: {
        labels: { color: textColor, font: { family: "'Inter', sans-serif", size: 12 } },
      },
    },
    scales: {
      x: {
        title: { display: true, text: "Step", color: textColor },
        grid: { color: gridColor },
        ticks: { color: textColor },
      },
      y: {
        grid: { color: gridColor },
        ticks: { color: textColor },
      },
    },
  };

  // Loss chart - SFT and GRPO loss curves
  const sftSteps = Array.from({ length: 20 }, (_, i) => (i + 1) * 25);
  const sftLoss = [
    2.45, 2.12, 1.89, 1.72, 1.58, 1.47, 1.38, 1.31, 1.25, 1.2, 1.16, 1.13,
    1.1, 1.08, 1.06, 1.04, 1.03, 1.02, 1.01, 1.0,
  ];
  const grpoSteps = Array.from({ length: 25 }, (_, i) => (i + 1) * 10);
  const grpoLoss = [
    1.85, 1.72, 1.61, 1.53, 1.46, 1.41, 1.37, 1.33, 1.3, 1.27, 1.24, 1.22,
    1.2, 1.18, 1.16, 1.15, 1.13, 1.12, 1.11, 1.1, 1.09, 1.08, 1.08, 1.07,
    1.07,
  ];

  const lossCtx = document.getElementById("loss-chart");
  if (lossCtx) {
    new Chart(lossCtx, {
      type: "line",
      data: {
        labels: grpoSteps,
        datasets: [
          {
            label: "SFT Loss",
            data: sftLoss.concat(Array(5).fill(null)),
            borderColor: STAGE_COLORS.sft,
            backgroundColor: STAGE_COLORS.sft + "20",
            tension: 0.3,
            pointRadius: 0,
            borderWidth: 2,
          },
          {
            label: "GRPO Loss",
            data: grpoLoss,
            borderColor: STAGE_COLORS.grpo,
            backgroundColor: STAGE_COLORS.grpo + "20",
            tension: 0.3,
            pointRadius: 0,
            borderWidth: 2,
          },
        ],
      },
      options: {
        ...commonOptions,
        scales: {
          ...commonOptions.scales,
          y: {
            ...commonOptions.scales.y,
            title: { display: true, text: "Loss", color: textColor },
          },
        },
      },
    });
  }

  // Reward chart - GRPO reward over time
  const rewardSteps = Array.from({ length: 25 }, (_, i) => (i + 1) * 10);
  const rewardData = [
    0.05, 0.08, 0.12, 0.15, 0.19, 0.24, 0.28, 0.31, 0.35, 0.38, 0.41, 0.44,
    0.46, 0.48, 0.5, 0.52, 0.54, 0.55, 0.57, 0.58, 0.59, 0.6, 0.61, 0.61,
    0.62,
  ];

  const rewardCtx = document.getElementById("reward-chart");
  if (rewardCtx) {
    new Chart(rewardCtx, {
      type: "line",
      data: {
        labels: rewardSteps,
        datasets: [
          {
            label: "Mean Reward",
            data: rewardData,
            borderColor: STAGE_COLORS.grpo,
            backgroundColor: STAGE_COLORS.grpo + "20",
            fill: true,
            tension: 0.3,
            pointRadius: 0,
            borderWidth: 2,
          },
        ],
      },
      options: {
        ...commonOptions,
        scales: {
          ...commonOptions.scales,
          y: {
            ...commonOptions.scales.y,
            title: { display: true, text: "Reward", color: textColor },
            min: 0,
            max: 1,
          },
        },
      },
    });
  }
}

/* ── Init ───────────────────────────────────────────────────── */

document.addEventListener("DOMContentLoaded", () => {
  loadBenchmarks();
  renderCharts();
});
