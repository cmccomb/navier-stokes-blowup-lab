function setText(selector, value) {
  document.querySelector(selector).textContent = value;
}

function setMedia(selector, path) {
  document.querySelector(selector).setAttribute("src", path);
}

function setDownload(selector, path) {
  document.querySelector(selector).setAttribute("href", path);
}

function showRun(run, generatedAt) {
  setText("#run-resolution", `${run.resolution}³`);
  setText("#run-end", run.t_end.toFixed(3));
  setText("#rest-until", run.rest_until.toFixed(2));
  setText("#metric-peak", run.peak_speed.toFixed(3));
  setText("#metric-vorticity", run.peak_vorticity.toFixed(2));
  setText(
    "#metric-scale",
    `${Math.min(run.cells_per_radial_scale, run.cells_per_axial_scale).toFixed(2)} cells`,
  );

  setMedia("#axial-gif", run.media.axial_jet);
  setMedia("#swirl-gif", run.media.equatorial_swirl);
  setMedia("#forcing-gif", run.media.pulse_forcing);
  setDownload("#download-axial", run.media.axial_jet);
  setDownload("#download-swirl", run.media.equatorial_swirl);
  setDownload("#download-forcing", run.media.pulse_forcing);

  setText(
    "#data-status",
    `${run.frames} solver checkpoints · verified ${new Date(generatedAt).toLocaleDateString(undefined, {
      year: "numeric",
      month: "long",
      day: "numeric",
    })}`,
  );
}

async function loadResult() {
  try {
    const response = await fetch("data/results.json", { cache: "no-cache" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();
    const run =
      data.runs.find((candidate) => candidate.id === data.featured_id) ?? data.runs[0];
    showRun(run, data.generated_at);
  } catch (error) {
    setText("#data-status", "Published values shown · live record unavailable");
    console.error(error);
  }
}

loadResult();
