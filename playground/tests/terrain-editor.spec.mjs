import { test, expect } from "@playwright/test";

test("floating panels and terrain dimensions update the active MuJoCo scene", async ({ page }) => {
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto("/");
  await expect(page.locator("#start")).toBeEnabled({ timeout: 60_000 });

  for (const [toggle, panel] of [["cmdToggle", "cmdPanel"], ["monitorToggle", "monitorPanel"], ["terrainToggle", "terrainPanel"]]) {
    await page.locator(`#${toggle}`).click();
    const handle = page.locator(`#${panel} .panel-heading`);
    const before = await page.locator(`#${panel}`).boundingBox();
    const grip = await handle.boundingBox();
    const dx = before.x > 720 ? -100 : 100;
    await page.mouse.move(grip.x + 35, grip.y + 15);
    await page.mouse.down();
    await page.mouse.move(grip.x + 35 + dx, grip.y + 95, { steps: 5 });
    await page.mouse.up();
    const after = await page.locator(`#${panel}`).boundingBox();
    expect(Math.abs(after.x - before.x)).toBeGreaterThan(50);
    expect(after.y).toBeGreaterThan(before.y + 50);
    if (panel !== "terrainPanel") await page.locator(`#${panel} .icon-button`).click();
  }

  const map = page.locator("#terrainMap");
  const bounds = await map.boundingBox();
  const place = async (tool, x) => {
    await page.locator(`[data-terrain-tool="${tool}"]`).click();
    await map.click({ position: { x: bounds.width * x, y: bounds.height * .5 } });
  };
  const edit = async (id, value) => {
    await page.locator(`#${id}`).fill(String(value));
    await page.locator(`#${id}`).dispatchEvent("change");
  };
  await place("stairs", .16);
  await expect(page.locator("#stepCount")).toHaveValue("5");
  await edit("stepDepth", .4);
  await edit("stepWidth", 2);
  await edit("stepHeight", .2);
  await edit("stepCount", 4);
  await expect(page.locator(".terrain-element.selected")).toContainText("4 阶");

  await place("ramp", .49);
  await edit("elementSlope", 25);
  await expect(page.locator("#elementHeight")).toHaveValue("0.6");
  await edit("elementHeight", .72);
  await expect(page.locator("#elementSlope")).toHaveValue("30");

  await place("stones", .82);
  await edit("stoneColumns", 4);
  await edit("stoneRows", 3);
  await edit("stoneGapX", .2);
  await page.locator("#stoneLayout").selectOption("staggered");
  await expect(page.locator(".terrain-element.selected")).toContainText("4 × 3 株");
  await page.locator("#terrainApply").click();
  await expect(page.locator("#engineState")).toHaveText("浏览器物理引擎已就绪", { timeout: 30_000 });
  expect(errors).toEqual([]);
});

test("dragging terrain map markers moves existing obstacles without adding another", async ({ page }) => {
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto("/");
  await expect(page.locator("#terrainToggle")).toBeEnabled({ timeout: 60_000 });
  await page.locator("#terrainToggle").click();
  await page.locator('[data-terrain-tool="ramp"]').click();
  const map = page.locator("#terrainMap");
  const bounds = await map.boundingBox();
  await map.click({ position: { x: bounds.width * .25, y: bounds.height * .5 } });
  await map.click({ position: { x: bounds.width * .6, y: bounds.height * .5 } });
  await expect(page.locator(".terrain-element")).toHaveCount(2);
  const firstX = bounds.x + bounds.width * .25, markerY = bounds.y + bounds.height * .5;
  await page.mouse.move(firstX, markerY);
  await page.mouse.down();
  await page.mouse.move(firstX + 20, markerY, { steps: 4 });
  await page.mouse.up();
  await expect(page.locator(".terrain-element.selected")).toContainText("1. 斜坡");
  expect(Number(await page.locator("#elementX").inputValue())).toBeGreaterThan(-4.5);
  await expect(page.locator(".terrain-element")).toHaveCount(2);

  const secondX = bounds.x + bounds.width * .6;
  await page.keyboard.down("Shift");
  await page.mouse.move(secondX, markerY);
  await page.mouse.down({ button: "right" });
  await page.mouse.move(secondX - 20, markerY, { steps: 4 });
  await page.mouse.up({ button: "right" });
  await page.keyboard.up("Shift");
  await expect(page.locator(".terrain-element.selected")).toContainText("2. 斜坡");
  expect(Number(await page.locator("#elementX").inputValue())).toBeLessThan(1.5);
  await expect(page.locator(".terrain-element")).toHaveCount(2);
  await page.locator("#terrainApply").click();
  await expect(page.locator("#engineState")).toHaveText("浏览器物理引擎已就绪", { timeout: 30_000 });
  expect(errors).toEqual([]);
});
