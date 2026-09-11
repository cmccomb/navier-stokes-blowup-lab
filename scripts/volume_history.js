/* One selected saved volume, at most three cached frames, no time-series stack. */
(() => {
  "use strict";
  const history = JSON.parse(document.getElementById("volume-history").textContent);
  const byId = id => document.getElementById(id);
  const plot = byId("stream-volume"), slider = byId("timeline");
  const play = byId("play"), status = byId("status"), component = byId("component");
  const cache = new Map(), count = history.frames.length, axis = history.axis, n = axis.length;
  let shown = -1, wanted = 0, busy = false, playing = false, timer = null, initialized = false;
  let selection = 0, renderedSelection = -1, stopped = false;
  const x = [], y = [], z = [];
  for (const a of axis) for (const b of axis) for (const c of axis) { x.push(a); y.push(b); z.push(c); }
  const field = history.field === "velocity" ? "Velocity" : "Applied forcing";
  const symbol = history.field === "velocity" ? "u" : "f";
  // Plotly.js does not resolve every Python colorscale name; pin the same Magma stops.
  const magma = ["#000004","#180f3d","#440f76","#721f81","#9e2f7f","#cd4071","#f1605d","#fd9668","#feca8d","#fcfdbf"].map((color,i) => [i/9,color]);
  byId("title").textContent = `${field} · full saved 3D history`;
  byId("sampling").textContent = `${n}³ browser samples from ${history.source_resolution}³ saved fields. All ${count} frames; loaded on demand. Native archives retained.`;
  slider.max = count - 1;

  async function load(index) {
    if (cache.has(index)) {
      const value = cache.get(index); cache.delete(index); cache.set(index, value); return value;
    }
    const frame = history.frames[index];
    const url = new URL(frame.path.replace(/^site\/media\//, ""), location.href);
    const response = await fetch(url, {cache: "force-cache", signal: AbortSignal.timeout(30000)});
    if (!response.ok) throw new Error(`Frame download failed (${response.status}).`);
    const packed = await response.arrayBuffer();
    if (packed.byteLength !== frame.bytes) throw new Error("Incomplete frame download.");
    const hash = Array.from(new Uint8Array(await crypto.subtle.digest("SHA-256", packed)), v => v.toString(16).padStart(2, "0")).join("");
    if (hash !== frame.sha256) throw new Error("Frame checksum mismatch.");
    const stream = new Blob([packed]).stream().pipeThrough(new DecompressionStream("gzip"));
    const raw = await new Response(stream).arrayBuffer();
    if (raw.byteLength !== n * n * n * 3 * 4) throw new Error("Unexpected frame dimensions.");
    // DataView makes the little-endian archive contract explicit on every host.
    const data = new DataView(raw), values = new Float32Array(raw.byteLength / 4);
    for (let i = 0; i < values.length; i++) {
      values[i] = data.getFloat32(i * 4, true);
      if (!Number.isFinite(values[i])) throw new Error("Nonfinite saved vector.");
    }
    cache.set(index, values);
    while (cache.size > history.cache_frames) cache.delete(cache.keys().next().value);
    plot.dataset.cachedFrames = cache.size;
    return values;
  }

  function traces(vectors, selected) {
    const magnitude = selected === "magnitude", limit = history.limits[selected];
    const slot = {x: 0, y: 1, z: 2}[selected], value = new Float64Array(n ** 3);
    for (let i = 0; i < value.length; i++) value[i] = magnitude
      ? Math.hypot(vectors[3 * i], vectors[3 * i + 1], vectors[3 * i + 2]) : vectors[3 * i + slot];
    const cone = {type:"cone", x:[], y:[], z:[], u:[], v:[], w:[], showscale:false,
      colorscale:"Viridis", cauto:false, cmin:0, cmax:history.limits.magnitude,
      sizemode:"raw", sizeref:0.35 * history.half_domain / history.limits.magnitude, anchor:"tail"};
    const stride = Math.ceil(n / 9);
    for (let i = 0; i < n; i += stride) for (let j = 0; j < n; j += stride) for (let k = 0; k < n; k += stride) {
      const offset = ((i * n + j) * n + k) * 3;
      const u = vectors[offset], v = vectors[offset + 1], w = vectors[offset + 2];
      if (Math.hypot(u, v, w) <= 0.005 * history.limits.magnitude) continue;
      cone.x.push(axis[i]); cone.y.push(axis[j]); cone.z.push(axis[k]);
      cone.u.push(u); cone.v.push(v); cone.w.push(w);
    }
    return [{type:"isosurface", x, y, z, value, isomin:(magnitude ? 0.02 : -0.8)*limit,
      isomax:0.8*limit, cmin:magnitude ? 0 : -limit, cmax:limit, cauto:false,
      surface:{count:magnitude ? 3 : 6}, colorscale:magnitude ? magma : "RdBu",
      reversescale:!magnitude, opacity:0.25, caps:{x:{show:false},y:{show:false},z:{show:false}},
      colorbar:{title:{text:magnitude ? `|${symbol}|` : `${symbol}_${selected}`}, thickness:12, len:0.65, x:1.01},
      showscale:true}, cone];
  }

  function layout() {
    const half = history.half_domain;
    const coordinate = title => ({title:{text:title}, range:[-half,half], autorange:false,
      tickfont:{size:13}, nticks:5, backgroundcolor:"#07111f", gridcolor:"#33475a", zerolinecolor:"#63778c"});
    return {paper_bgcolor:"#07111f", font:{family:"Arial, Helvetica, sans-serif", color:"#e9f1f5", size:14},
      margin:{l:12,r:64,t:32,b:32}, showlegend:false, uirevision:"preserve-camera",
      scene:{xaxis:coordinate("x"), yaxis:coordinate("y"), zaxis:coordinate("z"), aspectmode:"cube",
        bgcolor:"#07111f", camera:{eye:{x:2.8,y:2.65,z:1.9}}, uirevision:"preserve-camera"}};
  }

  function pause() { playing = false; clearTimeout(timer); play.textContent = shown === count - 1 ? "Replay" : "Play"; }
  function request(index) {
    wanted = Math.max(0, Math.min(count - 1, index)); selection++; slider.value = wanted;
    clearTimeout(timer); byId("retry").hidden = true; void pump();
  }
  async function pump() {
    if (busy || stopped) return;
    busy = true;
    try {
      while (renderedSelection !== selection) {
        const ticket = selection, index = wanted, selected = component.value;
        status.className = "";
        status.textContent = `Loading frame ${index + 1} / ${count}…`;
        const vectors = await load(index);
        if (ticket !== selection) continue;
        const data = traces(vectors, selected);
        if (!initialized) {
          await Plotly.newPlot(plot, data, layout(), {responsive:true, displaylogo:false});
          initialized = true;
        } else {
          // Restyle only: never reset camera, axis range, or zoom while scrubbing.
          await Plotly.restyle(plot, Object.fromEntries(Object.entries(data[0]).filter(([key]) => key !== "type").map(([key,v]) => [key,[v]])), [0]);
          await Plotly.restyle(plot, Object.fromEntries(Object.entries(data[1]).filter(([key]) => key !== "type").map(([key,v]) => [key,[v]])), [1]);
        }
        shown = index; renderedSelection = ticket;
        plot.dataset.frameIndex = index; plot.dataset.component = selected;
        const t = history.frames[index].time;
        byId("position").textContent = `Frame ${index + 1} / ${count} · t = ${t.toFixed(6)}${t === 0 ? " · at rest" : ""}`;
        slider.setAttribute("aria-valuetext", `Frame ${index + 1} of ${count}, time ${t.toFixed(6)}`);
        status.textContent = index === 0 && t === 0 ? "Zero field: no surfaces or arrows at rest." : "Drag to rotate · scroll to zoom";
        for (const id of ["play", "first", "last", "timeline"]) byId(id).disabled = false;
      }
      if (playing && shown < count - 1) timer = setTimeout(() => request(shown + 1), 650);
      else if (shown === count - 1) pause();
    } catch (error) {
      pause(); status.className = "error";
      status.textContent = `${error.message} ${shown < 0 ? "No frame displayed." : `Still showing frame ${shown + 1}.`}`;
      byId("retry").hidden = false;
    } finally { busy = false; }
  }
  play.addEventListener("click", () => {
    if (playing) { pause(); return; }
    playing = true; play.textContent = "Pause";
    request(shown === count - 1 ? 0 : Math.max(shown, 0));
  });
  slider.addEventListener("input", () => { pause(); request(Number(slider.value)); });
  component.addEventListener("change", () => { pause(); request(wanted); });
  byId("first").addEventListener("click", () => { pause(); request(0); });
  byId("last").addEventListener("click", () => { pause(); request(count - 1); });
  byId("retry").addEventListener("click", () => request(wanted));
  window.addEventListener("pagehide", () => { pause(); stopped = true; });
  if (!("DecompressionStream" in window) || !crypto.subtle) {
    status.className = "error"; status.textContent = "This 3D viewer needs a current browser with gzip decompression and secure HTTPS. The 2D videos remain available.";
  } else request(0);
})();
