const percent = (value) => `${(100 * value).toFixed(3)}%`;
const scientific = (value) => value.toExponential(2).replace("e-", "e−");

function setText(selector, value) {
  document.querySelector(selector).textContent = value;
}

function showRun(run, generatedAt) {
  setText("#run-resolution", `${run.resolution}³`);
  setText("#run-end", run.t_end.toFixed(2));
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
  setText(
    "#data-status",
    `Verified ${new Date(generatedAt).toLocaleDateString(undefined, {
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
    showRun(data.run, data.generated_at);
  } catch (error) {
    setText("#data-status", "Published values shown · live record unavailable");
    console.error(error);
  }
}

loadResult();
