const formatPercent = (value, digits = 3) => `${(100 * value).toFixed(digits)}%`;
const formatScientific = (value) => value.toExponential(2).replace("e-", "e−");

function updateHeadline(run) {
  document.querySelector("#metric-resolution").textContent = `${run.resolution}³`;
  document.querySelector("#metric-time").textContent = `at t = ${run.t_end}`;
  document.querySelector("#metric-vorticity").textContent = run.peak_vorticity.toFixed(2);
  document.querySelector("#metric-tracking").textContent = formatPercent(run.tracking_relative_l2);
  document.querySelector("#metric-tail").textContent = formatPercent(run.spectral_tail_fraction);
  document.querySelector("#metric-divergence").textContent = formatScientific(run.divergence_linf);
}

async function loadResults() {
  const status = document.querySelector("#dataset-status");
  try {
    const response = await fetch("data/results.json", { cache: "no-cache" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();
    updateHeadline(data.headline);
    status.textContent = `Verified ${new Date(data.generated_at).toLocaleDateString(undefined, {
      year: "numeric",
      month: "short",
      day: "numeric",
    })} · current best of ${data.history_count} recorded endpoints`;
  } catch (error) {
    status.textContent = "The current-best record could not be loaded.";
    console.error(error);
  }
}

loadResults();
