const refreshInterval = 60_000;
let shownRevision = null;
let lastObservedAt = null;

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
  for (const kind of ["flow", "force"]) {
    if (!document.querySelector(`#${kind}-video`).error) {
      document.querySelector(`#${kind}-pending`).hidden = true;
    }
  }
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
  for (const kind of ["flow", "force"]) {
    const path = run.media[`${kind}_3d`];
    const tab = document.querySelector(`#${kind}-tab-3d`);
    tab.disabled = !path;
    if (path) {
      const url = mediaUrl(path, run.revision);
      const iframe = document.querySelector(`#${kind}-3d`);
      iframe.dataset.latestSrc = url;
      document.querySelector(`#${kind}-3d-link`).href = url;
      document.querySelector(`#${kind}-refresh-3d`).hidden = !iframe.hasAttribute("src") || iframe.src === url;
    }
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
    for (const kind of ["flow", "force"]) {
      const pending = document.querySelector(`#${kind}-pending`);
      pending.textContent = lastObservedAt
        ? `Update unavailable. Showing the clip observed ${dateTime(lastObservedAt)}.`
        : "Current movie unavailable. It will appear when published.";
      pending.hidden = false;
    }
    console.error(error);
  } finally {
    window.setTimeout(loadResult, refreshInterval);
  }
}

const resume2d = new Map();

function load3d(kind) {
  const iframe = document.querySelector(`#${kind}-3d`);
  if (iframe.dataset.latestSrc && iframe.src !== iframe.dataset.latestSrc) {
    iframe.src = iframe.dataset.latestSrc;
  }
  document.querySelector(`#${kind}-refresh-3d`).hidden = true;
}

function selectView(kind, view) {
  if (document.querySelector(`#${kind}-tab-${view}`).getAttribute("aria-selected") === "true") return;
  for (const option of ["2d", "3d"]) {
    const active = option === view;
    const tab = document.querySelector(`#${kind}-tab-${option}`);
    tab.setAttribute("aria-selected", String(active));
    tab.tabIndex = active ? 0 : -1;
    document.querySelector(`#${kind}-panel-${option}`).hidden = !active;
  }
  const video = document.querySelector(`#${kind}-video`);
  if (view === "3d") {
    resume2d.set(kind, !video.paused);
    video.pause();
    load3d(kind);
  } else if (resume2d.get(kind) && !window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
    video.play().catch(() => {});
  }
}

for (const tab of document.querySelectorAll("[role=tab]")) {
  tab.addEventListener("click", () => selectView(tab.dataset.field, tab.dataset.view));
  tab.addEventListener("keydown", (event) => {
    if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
    event.preventDefault();
    const view = event.key === "Home" ? "2d" : event.key === "End" ? "3d" : tab.dataset.view === "2d" ? "3d" : "2d";
    const next = document.querySelector(`#${tab.dataset.field}-tab-${view}`);
    if (!next.disabled) {
      selectView(tab.dataset.field, view);
      next.focus();
    }
  });
}
for (const button of document.querySelectorAll("[data-refresh-volume]")) {
  button.addEventListener("click", () => load3d(button.dataset.refreshVolume));
}

loadResult();
