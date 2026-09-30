import { test, expect } from "@playwright/test";

test("Go2 switches to recovery after a fall and resumes the selected policy", async ({ page }) => {
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto("http://127.0.0.1:4176/");
  await expect(page.locator("#start")).toBeEnabled({ timeout: 60_000 });
  await page.selectOption("#policy", "trot");
  await expect(page.locator("#engineState")).toHaveText("浏览器物理引擎已就绪");
  await page.locator("#start").click();

  await page.evaluate(() => window.__playgroundDebug.setBasePose(.18, [.70710678, .70710678, 0, 0]));
  await expect.poll(() => page.evaluate(() => window.__playgroundDebug.recoveryState().active), { timeout: 20_000 }).toBe(true);
  await expect(page.locator("#engineState")).toHaveText(/正在运行起身策略|正在确认站稳/);
  await expect(page.locator("#policy")).toHaveValue("trot");
  await expect.poll(() => page.evaluate(() => window.__playgroundDebug.recoveryState().action.some((value) => value !== 0))).toBe(true);
  await expect.poll(() => page.evaluate(() => window.__playgroundDebug.recoveryState().active), { timeout: 30_000 }).toBe(false);
  await page.locator("#start").click();
  await expect(page.locator("#engineState")).toContainText("继续 小跑");
  await expect(page.locator("#policy")).toHaveValue("trot");
  const completed = await page.evaluate(() => window.__playgroundDebug.recoveryState());
  expect(completed.phase).toBe("normal");
  const events = completed.transitions.map(({ event }) => event);
  expect(events[0]).toBe("recover");
  expect(events).toContain("stabilize");
  expect(events.at(-1)).toBe("resume");
  const resumed = completed.transitions.at(-1);
  expect(resumed.sensors.supportFeet).toBeGreaterThanOrEqual(2);
  expect(resumed.sensors.bodyContact).toBe(false);
  expect(completed.action.every((value) => value === 0)).toBe(true);

  // A later fall must still trigger, and reset during recovery must clear both
  // confirmation progress and the status shown to the user.
  await page.locator("#reset").click();
  await expect(page.locator("#reset")).toBeEnabled();
  await page.evaluate(() => window.__playgroundDebug.setBasePose(.18, [.70710678, -.70710678, 0, 0]));
  await page.locator("#start").click();
  await expect.poll(() => page.evaluate(() => window.__playgroundDebug.recoveryState().active), { timeout: 20_000 }).toBe(true);
  await page.locator("#reset").click();
  await expect(page.locator("#reset")).toBeEnabled();
  await page.locator("#start").click();
  const reset = await page.evaluate(() => window.__playgroundDebug.recoveryState());
  expect(reset.active).toBe(false);
  expect(reset.phase).toBe("normal");
  expect(reset.uprightTime).toBe(0);
  expect(reset.transitions).toEqual([]);
  await expect(page.locator("#engineState")).toHaveText("浏览器物理引擎已就绪");
  await expect(page.locator("#engineState")).not.toHaveClass(/error/);
  expect(errors).toEqual([]);
});

test("back-down recovery returns to a history policy with real foot support", async ({ page }) => {
  await page.goto("http://127.0.0.1:4176/");
  await expect(page.locator("#start")).toBeEnabled({ timeout: 60_000 });
  await page.selectOption("#policy", "arenaWalk");
  await expect(page.locator("#engineState")).toHaveText("浏览器物理引擎已就绪");
  await page.evaluate(() => window.__playgroundDebug.setBasePose(.15, [0, 1, 0, 0]));
  await page.locator("#start").click();
  await expect.poll(() => page.evaluate(() => window.__playgroundDebug.recoveryState().active), { timeout: 20_000 }).toBe(true);
  await expect.poll(() => page.evaluate(() => window.__playgroundDebug.recoveryState().transitions.some(({ event }) => event === "resume")), { timeout: 30_000 }).toBe(true);
  await page.locator("#start").click();
  const state = await page.evaluate(() => window.__playgroundDebug.recoveryState());
  const resumed = state.transitions.find(({ event }) => event === "resume");
  expect(resumed.sensors.supportFeet).toBeGreaterThanOrEqual(2);
  expect(resumed.sensors.bodyContact).toBe(false);
  expect(state.selected).toBe("arenaWalk");
  expect(state.active).toBe(false);
  await expect(page.locator("#engineState")).toContainText("继续 Arena 平地行走");
});
