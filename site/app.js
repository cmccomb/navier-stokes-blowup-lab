const percent = (value) => `${(100 * value).toFixed(3)}%`;

function setText(selector, value) {
  document.querySelector(selector).textContent = value;
}

function showRun(run, generatedAt) {
  setText("#run-label", run.label);
  setText("#run-resolution", `${run.resolution}³`);
  setText("#run-end", run.t_end.toFixed(3));
  setText("#run-frames", run.frames);
  setText("#rest-until", run.rest_until.toFixed(2));
  setText("#metric-peak", run.peak_speed.toFixed(3));
  setText("#metric-vorticity", run.peak_vorticity.toFixed(2));
  setText("#metric-tracking", percent(run.tracking_relative_l2));

  const video = document.querySelector("#result-video");
  const source = document.querySelector("#result-video-source");
  if (source.getAttribute("src") !== run.media.mp4) {
    source.setAttribute("src", run.media.mp4);
    video.setAttribute("poster", run.media.poster);
    video.load();
  }
  video.setAttribute(
    "aria-label",
    `Computed and target velocity from rest through t equals ${run.t_end.toFixed(3)}`,
  );
  document.querySelector("#download-mp4").setAttribute("href", run.media.mp4);
  document.querySelector("#download-gif").setAttribute("href", run.media.gif);
  setText("#media-title", `${run.label}: PhiFlow versus target`);
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
    const run =
      data.runs.find((candidate) => candidate.id === data.featured_id) ??
      data.runs[0];
    showRun(run, data.generated_at);
  } catch (error) {
    setText("#data-status", "Published values shown · live record unavailable");
    console.error(error);
  }
}

loadResult();
