const refreshInterval = 60_000;
let shownRevision = null;
let lastObservedAt = null;

function setText(selector, value) {
  document.querySelector(selector).textContent = value;
}

function number(value, digits = 3) {
  return Number.isFinite(value) ? value.toFixed(digits) : "—";
}

function dateTime(value) {
  return new Date(value).toLocaleString(undefined, {
    year: "numeric", month: "short", day: "numeric",
    hour: "numeric", minute: "2-digit", timeZoneName: "short",
  });
}

function mediaUrl(path, revision) {
  const url = new URL(path, window.location.href);
  if (url.origin !== window.location.origin || !url.pathname.includes("/media/")) {
    throw new Error("Expected a local published media path");
  }
  url.searchParams.set("v", revision);
  return url.href;
}

function showMedia(kind, media, revision) {
  const video = document.querySelector(`#${kind}-video`);
  const firstLoad = !video.hasAttribute("src");
  const continuePlaying = firstLoad || !video.paused;
  const mp4 = mediaUrl(media[`${kind}_mp4`], revision);
  const gif = mediaUrl(media[`${kind}_gif`], revision);
  document.querySelector(`#${kind}-gif`).href = gif;
  document.querySelector(`#${kind}-mp4`).href = mp4;
  document.querySelector(`#${kind}-figure`).hidden = false;
  document.querySelector(`#${kind}-pending`).hidden = true;
  video.onloadeddata = () => {
    document.querySelector(`#${kind}-pending`).hidden = true;
  };
  video.onerror = () => {
    const pending = document.querySelector(`#${kind}-pending`);
    pending.textContent = "Movie unavailable. Try the GIF download or the next published update.";
    pending.hidden = false;
  };
  video.src = mp4;
  video.muted = true;
  if (continuePlaying && !window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
    video.play().catch(() => {}); // Controls remain available if autoplay is blocked.
  }
}

function showRun(run) {
  const { config, diagnostics } = run;
  setText("#run-state", {
    running: "in progress at last update",
    complete: "complete",
    stopped: "stopped · partial run",
  }[run.status]);
  setText("#run-resolution", `${config.resolution}³`);
  setText("#grid-resolution", Array(3).fill(config.resolution).join(" × "));
  setText("#grid-cells", `${(config.resolution ** 3 / 1e6).toFixed(2)} million cells`);
  setText("#display-resolution", `${run.display_resolution}³`);
  setText("#run-end", number(config.t_end));
  setText("#rest-until", number(config.paper_time_cutoff_start, 2));
  setText("#latest-time", number(run.latest_t, 5));
  setText("#frame-status", `${run.captured_frames} movie frames captured · ${run.clip_times.length} in the current clip`);
  setText("#diagnostics-time", diagnostics
    ? `Diagnostics recorded at t = ${number(diagnostics.t, 5)}. These may precede the latest movie frame.`
    : "No diagnostic checkpoint published yet. Movie frames can arrive first.");
  setText("#metric-peak", number(diagnostics?.peak_speed));
  setText("#metric-vorticity", number(diagnostics?.peak_vorticity, 2));
  const radial = diagnostics?.cells_per_radial_scale;
  const axial = diagnostics?.cells_per_axial_scale;
  setText("#metric-scale", Number.isFinite(radial) && Number.isFinite(axial)
    ? `${number(Math.min(radial, axial), 2)} cells` : "—");
  setText("#scope-warning", run.scope_warning);
  setText("#data-status", `Run observed ${dateTime(run.observed_at)}.`);

  const times = run.clip_times;
  const range = times.length === 1
    ? `one saved frame at t = ${number(times[0], 5)} (held)`
    : `saved t = ${number(times[0], 5)} to ${number(times.at(-1), 5)}`;
  document.querySelectorAll("[data-clip-range]").forEach((node) => {
    node.textContent = range;
  });
  if (shownRevision !== run.revision) {
    showMedia("flow", run.media, run.revision);
    showMedia("force", run.media, run.revision);
    shownRevision = run.revision;
  }
  lastObservedAt = run.observed_at;
}

async function loadResult() {
  try {
    const response = await fetch("data/stream.json", { cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const run = await response.json();
    if (run.schema_version !== 1 || !["running", "complete", "stopped"].includes(run.status)) {
      throw new Error("Unsupported stream record");
    }
    showRun(run);
  } catch (error) {
    setText("#data-status", lastObservedAt
      ? `Update unavailable. Showing the saved record observed ${dateTime(lastObservedAt)}.`
      : "Current run record unavailable. Movies and measured values will appear when it is published.");
    if (!lastObservedAt) setText("#run-state", "awaiting published record");
    console.error(error);
  } finally {
    window.setTimeout(loadResult, refreshInterval);
  }
}

loadResult();
