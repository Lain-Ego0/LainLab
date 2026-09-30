import { test, expect } from "@playwright/test";

test("G1 DWAQ switches from AMP, climbs stairs, resets phase and switches back", async ({ page }) => {
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  const ready = async () => {
    await expect(page.locator("#start")).toBeEnabled({ timeout: 60_000 });
    await expect(page.locator("#engineState")).toHaveText("浏览器物理引擎已就绪");
  };
  await page.goto("/");
  await ready();
  await page.selectOption("#robotModel", "g1");
  await ready();
  await expect(page.locator("#policy option")).toHaveText(["G1 · AMP 平地行走", "G1 · DWAQ 越障"]);

  // A missing DWAQ artifact must leave the selected AMP session usable.
  await page.route("**/policies/g1-dwaq-phase.onnx", (route) => route.fulfill({ status: 404, body: "missing" }));
  await page.selectOption("#policy", "g1DwaqPhase");
  await expect(page.locator("#policy")).toBeEnabled();
  await expect(page.locator("#policy")).toHaveValue("g1Amp");
  await expect(page.locator("#engineState")).toHaveClass(/error/);
  await page.unroute("**/policies/g1-dwaq-phase.onnx");

  await page.locator("#start").click();
  await page.selectOption("#policy", "g1DwaqPhase");
  await ready();
  await expect(page.locator("#clock")).toHaveText("0.00 s");
  await expect(page.locator("#policyMeta")).toContainText("500");
  await expect(page.locator("#actionValues span")).toHaveCount(29);
  await expect(page.locator("#commandSpeedVx")).toHaveAttribute("max", "0.7");

  await page.locator("#terrainToggle").click();
  await page.locator("#terrainImport").setInputFiles({ name: "dwaq-stairs.json", mimeType: "application/json",
    buffer: Buffer.from(JSON.stringify({ version: 2, terrain: { kind: "flat", height: .28, seed: 7 },
      elements: [{ kind: "stairs", x: 2.93, y: 0, yaw: 0, stepDepth: .62, stepWidth: 3.6, stepHeight: .06, stepCount: 4 }] })) });
  await expect(page.locator(".terrain-element")).toHaveCount(1);
  await page.locator("#terrainApply").click();
  await ready();
  await page.locator("#terrainClose").click();
  await page.locator("#cmdToggle").click();
  await page.locator("#commandSpeedVx").evaluate((input) => { input.value = "0.3"; });
  await page.locator("#commandSpeedVx").dispatchEvent("input");
  await page.locator("#cmdClose").click();

  await page.evaluate(() => {
    window.__dwaqTrace = [];
    const sample = () => {
      const time = parseFloat(document.getElementById("clock").textContent);
      const xyz = document.getElementById("bodyPosition").textContent.match(/-?\d+\.\d+/g)?.map(Number);
      if (xyz && time > (window.__dwaqTrace.at(-1)?.time ?? -1)) window.__dwaqTrace.push({ time, xyz });
      window.__dwaqTraceFrame = requestAnimationFrame(sample);
    };
    sample();
  });
  await page.locator("#start").click();
  await page.keyboard.down("w");
  await expect.poll(async () => parseFloat(await page.locator("#clock").textContent()), { timeout: 60_000 }).toBeGreaterThan(16);
  await page.keyboard.up("w");
  await page.locator("#start").click();
  const trace = await page.evaluate(() => { cancelAnimationFrame(window.__dwaqTraceFrame); return window.__dwaqTrace; });
  const moving = trace.filter(({ time }) => time > 1);
  expect(Math.min(...moving.map(({ xyz }) => xyz[2]))).toBeGreaterThan(.55);
  expect(Math.max(...moving.map(({ xyz }) => xyz[0]))).toBeGreaterThan(4.3);
  const onStairs = moving.filter(({ xyz }) => xyz[0] > 3.4 && xyz[0] < 4.1 && Math.abs(xyz[1]) < 1.6);
  expect(onStairs.length).toBeGreaterThan(0);
  expect(Math.max(...onStairs.map(({ xyz }) => xyz[2]))).toBeGreaterThan(.94);
  await expect(page.locator("#engineState")).not.toHaveClass(/error/);

  await page.locator("#reset").click();
  await expect(page.locator("#reset")).toBeEnabled();
  await expect(page.locator("#clock")).toHaveText("0.00 s");
  await expect(page.locator("#policy")).toHaveValue("g1DwaqPhase");
  await page.locator("#start").click();
  await expect.poll(async () => parseFloat(await page.locator("#clock").textContent())).toBeGreaterThan(.5);
  await page.selectOption("#policy", "g1Amp");
  await ready();
  await expect(page.locator("#policyMeta")).toContainText("384");
  await expect(page.locator("#clock")).toHaveText("0.00 s");
  await page.locator("#start").click();
  await expect.poll(async () => parseFloat(await page.locator("#clock").textContent())).toBeGreaterThan(.5);
  await page.locator("#start").click();
  await expect(page.locator("#engineState")).not.toHaveClass(/error/);
  expect(errors).toEqual([]);
});
