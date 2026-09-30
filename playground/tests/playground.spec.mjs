import { test, expect } from "@playwright/test";

test("robot switching, G1 AMP, live reset, terrain and failed-load recovery", async ({ page }) => {
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto("/");
  const ready = async () => {
    await expect(page.locator("#start")).toBeEnabled({ timeout: 60_000 });
    await expect(page.locator("#engineState")).toHaveText("浏览器物理引擎已就绪");
  };
  const run = async () => {
    await page.locator("#start").click();
    await expect.poll(async () => parseFloat(await page.locator("#clock").textContent())).toBeGreaterThan(.2);
    await page.locator("#start").click();
    await expect(page.locator("#engineState")).not.toHaveClass(/error/);
  };
  await ready();
  await expect(page.locator("#policy option")).toHaveCount(8);
  await run();
  // Switch while an ONNX/physics step can be in flight.
  await page.locator("#start").click();
  await page.selectOption("#robotModel", "g1");
  await ready();
  await expect(page.locator("#policy option")).toHaveCount(1);
  await expect(page.locator("#policy")).toHaveValue("g1Amp");
  await expect(page.locator("#actionValues span")).toHaveCount(29);
  await expect(page.locator("#policy option")).toHaveText(["G1 · AMP 平地行走"]);
  await expect(page.locator("#clock")).toHaveText("0.00 s");
  await run();
  await page.locator("#start").click();
  await page.locator("#reset").click();
  await expect(page.locator("#reset")).toBeEnabled();
  await expect(page.locator("#start")).toHaveText("暂停");
  await page.locator("#start").click();
  await expect(page.locator("#commandSpeedVx")).toBeEnabled();
  // Recompile the active G1 scene with the same flat terrain, then infer again.
  await page.locator("#terrainToggle").click();
  await page.locator("#terrainApply").click();
  await ready();
  await run();
  await page.locator("#terrainClose").click();
  await page.selectOption("#robotModel", "go2");
  await ready();
  await expect(page.locator("#actionValues span")).toHaveCount(12);
  await page.selectOption("#policy", "trot");
  await ready();
  await run();
  // Failed G1 loads keep Go2 usable and restore the robot/policy selectors.
  await page.route("**/policies/g1-amp.onnx", (route) => route.fulfill({ status: 404, body: "missing" }));
  await page.selectOption("#robotModel", "g1");
  await expect(page.locator("#robotModel")).toBeEnabled();
  await expect(page.locator("#robotModel")).toHaveValue("go2");
  await expect(page.locator("#policy")).toHaveValue("trot");
  await expect(page.locator("#engineState")).toHaveClass(/error/);
  await page.unroute("**/policies/g1-amp.onnx");
  await page.selectOption("#robotModel", "g1");
  await ready();
  await expect(page.locator("#policy")).toHaveValue("g1Amp");
  await run();
  await page.setViewportSize({ width: 390, height: 844 });
  for (const id of ["robotModel", "policy", "start", "reset", "terrainToggle"]) {
    const bounds = await page.locator(`#${id}`).boundingBox();
    expect(bounds.x).toBeGreaterThanOrEqual(0);
    expect(bounds.x + bounds.width).toBeLessThanOrEqual(390);
  }
  expect(errors).toEqual([]);
});
