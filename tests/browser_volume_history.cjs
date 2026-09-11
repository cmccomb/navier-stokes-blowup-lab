/* Run against a locally served export; PLAYWRIGHT_MODULE may point to a bundled runtime. */
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || "playwright");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const zlib = require("node:zlib");

(async () => {
  const [url, output] = process.argv.slice(2);
  fs.mkdirSync(output, {recursive:true});
  const browser = await chromium.launch({executablePath:process.env.CHROME_PATH, headless:true,
    args:["--enable-unsafe-swiftshader"]});
  try {
    const page = await browser.newPage({viewport:{width:960,height:950}});
    const errors = [], fetched = [];
    page.on("pageerror", error => errors.push(error.message));
    page.on("request", request => { if (request.url().includes(".f32.gz")) fetched.push(request.url()); });
    await page.goto(url);
    const ready = index => page.waitForFunction(i => document.querySelector("#stream-volume").dataset.frameIndex === String(i), index, {timeout:60000});
    await ready(0);
    const history = await page.locator("#volume-history").evaluate(el => JSON.parse(el.textContent));
    assert(history.frames.length > 24);
    assert.equal(history.frames[0].time, 0);
    assert.equal(fetched.length, 1, "first paint must not fetch the history");
    await page.screenshot({path:path.join(output,"initial.png"),fullPage:true});
    const seek = async index => {
      await page.locator("#timeline").evaluate((el,i) => {el.value=i;el.dispatchEvent(new Event("input",{bubbles:true}));}, index);
      await ready(index);
    };
    const middle = Math.floor(history.frames.length/2), last = history.frames.length-1;
    await seek(middle);
    const camera = {eye:{x:1.7,y:2.1,z:1.2},center:{x:0,y:0,z:0},up:{x:0,y:0,z:1},projection:{type:"perspective"}};
    await page.evaluate(camera => Plotly.relayout("stream-volume", {"scene.camera":camera}), camera);
    await seek(last);
    const frameURL = new URL(history.frames[last].path.replace(/^site\/media\//,""),url).href;
    const nativeBytes = zlib.gunzipSync(await (await page.request.get(frameURL)).body());
    for (const component of ["x","y","z","magnitude"]) {
      await page.locator("#component").selectOption(component);
      await page.waitForFunction(c => document.querySelector("#stream-volume").dataset.component === c, component);
      assert.deepEqual(await page.evaluate(() => document.querySelector("#stream-volume").layout.scene.camera), camera);
      const displayed = await page.evaluate(() => Array.from(document.querySelector("#stream-volume").data[0].value));
      const slot = {x:0,y:1,z:2}[component];
      for (let i=0;i<displayed.length;i++) {
        const expected = component === "magnitude"
          ? Math.hypot(nativeBytes.readFloatLE(12*i),nativeBytes.readFloatLE(12*i+4),nativeBytes.readFloatLE(12*i+8))
          : nativeBytes.readFloatLE(12*i+4*slot);
        assert.equal(displayed[i],expected,`component ${component}, voxel ${i}`);
      }
    }
    await page.screenshot({path:path.join(output,"latest-desktop.png"),fullPage:true});
    await page.setViewportSize({width:343,height:930});
    await page.waitForFunction(() => document.querySelector("#stream-volume")._fullLayout.width < 343);
    await page.screenshot({path:path.join(output,"latest-mobile.png"),fullPage:true});
    assert(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
    assert.deepEqual(await page.evaluate(() => document.querySelector("#stream-volume").layout.scene.camera), camera);
    // Rapid scrubbing is latest-wins, even while a prior frame loads/renders.
    await page.locator("#timeline").evaluate((el,last) => {
      for (const i of [1,2,3,last-1]) {el.value=i;el.dispatchEvent(new Event("input",{bubbles:true}));}
    },last);
    await ready(last-1);
    assert(Number(await page.locator("#stream-volume").getAttribute("data-cached-frames")) <= 3);
    await page.locator("#play").click();
    await ready(last);
    await page.waitForFunction(() => document.querySelector("#play").textContent === "Replay");
    // A missing uncached frame must not be silently skipped or relabeled.
    const bad = 5;
    const badUrl = new URL(history.frames[bad].path.replace(/^site\/media\//,""),url).href;
    await page.route(badUrl, route => route.fulfill({status:503,body:"test unavailable"}));
    await page.locator("#timeline").evaluate((el,i) => {el.value=i;el.dispatchEvent(new Event("input",{bubbles:true}));},bad);
    await page.locator("#retry").waitFor({state:"visible"});
    assert.equal(await page.locator("#stream-volume").getAttribute("data-frame-index"),String(last));
    assert.match(await page.locator("#status").textContent(),/Still showing/);
    await page.unroute(badUrl);
    await page.locator("#retry").click();
    await ready(bad);
    assert(Number(await page.locator("#stream-volume").getAttribute("data-cached-frames")) <= 3);
    assert.deepEqual(errors,[]);
    const embedded = await browser.newPage({viewport:{width:960,height:900}});
    await embedded.goto(new URL("../",url).href);
    await embedded.setContent(`<iframe src="${url}" sandbox="allow-scripts allow-downloads" style="width:920px;height:760px;border:0"></iframe>`);
    const frame = await (await embedded.$("iframe")).contentFrame();
    await frame.waitForFunction(() => document.querySelector("#stream-volume")?.dataset.frameIndex === "0",null,{timeout:60000});
    await frame.locator("#last").click();
    await frame.waitForFunction(i => document.querySelector("#stream-volume").dataset.frameIndex === String(i),last,{timeout:60000});
    await embedded.screenshot({path:path.join(output,"sandboxed-iframe.png"),fullPage:true});
    await embedded.close();
    console.log(JSON.stringify({frames:history.frames.length,first:0,last:history.frames[last].time,
      requestedChunks:fetched.length,cacheMax:3,errors,screenshots:output}));
  } finally { await browser.close(); }
})().catch(error => {console.error(error);process.exit(1);});
