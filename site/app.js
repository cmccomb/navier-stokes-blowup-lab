const percent = (value) => `${(100 * value).toFixed(3)}%`;
const scientific = (value) => value.toExponential(2).replace("e-", "e−");
const timestep = (value) => (value < 0.001 ? value.toExponential(2) : value.toPrecision(3));

function setText(selector, value) {
  document.querySelector(selector).textContent = value;
}

function showRun(run, generatedAt) {
  setText("#run-label", run.label);
  setText("#run-resolution", `${run.resolution}³`);
  setText("#run-end", run.t_end.toFixed(3));
  setText("#endpoint-time", run.t_end.toFixed(3));
  setText("#run-frames", run.frames);
  setText("#rest-until", run.rest_until.toFixed(2));
  setText("#metric-peak", run.peak_speed.toFixed(3));
  setText("#metric-vorticity", run.peak_vorticity.toFixed(2));
  setText("#metric-tracking", percent(run.tracking_relative_l2));
  setText("#radial-cells", run.cells_per_radial_scale.toFixed(2));
  setText("#axial-cells", run.cells_per_axial_scale.toFixed(2));
  setText("#spectral-tail", percent(run.spectral_tail_fraction));
  setText("#divergence", scientific(run.divergence_linf));
  setText("#kinetic-energy", run.kinetic_energy.toFixed(6));
  setText("#force-l2", run.force_l2.toFixed(2));
  setText("#max-dt", timestep(run.max_dt));
  setText("#cfl", run.cfl.toFixed(2));
  setText("#media-title", `${run.label}: computed field versus target`);

  const video = document.querySelector("#result-video");
  const source = document.querySelector("#result-video-source");
  const endpoint = document.querySelector("#endpoint-image");
  const mp4 = document.querySelector("#download-mp4");
  const gif = document.querySelector("#download-gif");
  if (source.getAttribute("src") !== run.media.mp4) {
    source.setAttribute("src", run.media.mp4);
    video.setAttribute("poster", run.media.poster);
    video.load();
  }
  video.setAttribute(
    "aria-label",
    `${run.label}, computed and target velocity from rest through t equals ${run.t_end.toFixed(3)}`,
  );
  endpoint.setAttribute("src", run.media.endpoint);
  endpoint.setAttribute(
    "alt",
    `Velocity slices for ${run.label} at rest and t equals ${run.t_end.toFixed(3)}`,
  );
  mp4.setAttribute("href", run.media.mp4);
  gif.setAttribute("href", run.media.gif);

  document.querySelectorAll(".run-tab").forEach((button) => {
    const active = button.dataset.runId === run.id;
    button.classList.toggle("is-active", active);
    button.setAttribute("aria-pressed", String(active));
  });
  setText(
    "#data-status",
    `Verified ${new Date(generatedAt).toLocaleDateString(undefined, {
      year: "numeric",
      month: "long",
      day: "numeric",
    })}`,
  );
}

function renderTabs(runs, selectRun) {
  const tabs = document.querySelector("#run-tabs");
  tabs.replaceChildren();
  for (const run of runs) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "run-tab";
    button.dataset.runId = run.id;
    button.setAttribute("aria-pressed", "false");
    button.textContent = run.label;
    button.addEventListener("click", () => selectRun(run));
    tabs.append(button);
  }
}

function renderComparison(runs) {
  const body = document.querySelector("#comparison-body");
  body.replaceChildren();
  for (const run of runs) {
    const row = document.createElement("tr");
    const label = document.createElement("th");
    label.scope = "row";
    label.textContent = run.label;
    row.append(label);
    for (const value of [
      `${run.resolution}³`,
      run.t_end.toFixed(3),
      timestep(run.max_dt),
      percent(run.tracking_relative_l2),
    ]) {
      const cell = document.createElement("td");
      cell.textContent = value;
      row.append(cell);
    }
    body.append(row);
  }
}

async function loadResults() {
  try {
    const response = await fetch("data/results.json", { cache: "no-cache" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();
    const featured =
      data.runs.find((run) => run.id === data.featured_id) ?? data.runs[0];
    const selectRun = (run) => showRun(run, data.generated_at);
    renderTabs(data.runs, selectRun);
    renderComparison(data.runs);
    selectRun(featured);
  } catch (error) {
    setText("#data-status", "Published values shown · live record unavailable");
    console.error(error);
  }
}

loadResults();
