import { test, expect } from "@playwright/test";
import { execFile } from "node:child_process";
import { promisify } from "node:util";
import { fileURLToPath } from "node:url";
import { readFile, stat } from "node:fs/promises";
import path from "node:path";

const exec = promisify(execFile);
const root = fileURLToPath(new URL("..", import.meta.url));

test("dev server still serves G1 assets after preparing them again", async ({ page, request }) => {
  const base = "http://127.0.0.1:4176";
  const scene = await readFile(path.join(root, "public/robot/g1/scene.xml"), "utf8");
  const assets = ["robot/scene_go2.xml", "robot/g1/scene.xml", "policies/g1-amp.onnx",
    ...[...scene.matchAll(/<mesh\b[^>]*\bfile="([^"]+)"/g)].map(([, file]) => `robot/g1/${file}`)];
  const expected = new Map();
  const timestamps = new Map();
  for (const asset of assets) {
    const file = path.join(root, "public", asset);
    const info = await stat(file);
    timestamps.set(asset, [info.ino, info.mtimeMs]);
    expected.set(asset, await readFile(file));
    const response = await request.get(`${base}/${asset}`);
    expect(response.ok()).toBeTruthy();
    expect(response.headers()["content-type"]).not.toContain("text/html");
    expect(await response.body()).toEqual(expected.get(asset));
  }
  // The dev server is already watching public/ when a build prepares assets.
  // Give delayed unlink/add events time to settle before checking its index.
  for (let run = 0; run < 3; run++) {
    await exec(process.execPath, ["scripts/prepare-assets.mjs"], { cwd: root });
    await page.waitForTimeout(1500);
    for (const asset of assets) {
      const info = await stat(path.join(root, "public", asset));
      expect([info.ino, info.mtimeMs]).toEqual(timestamps.get(asset));
      const response = await request.get(`${base}/${asset}`);
      expect(response.ok()).toBeTruthy();
      expect(response.headers()["content-type"]).not.toContain("text/html");
      expect(await response.body()).toEqual(expected.get(asset));
    }
  }
  const missing = await request.get(`${base}/robot/g1/meshes/missing.STL`);
  expect(missing.status()).toBe(404);
  expect(missing.headers()["content-type"]).not.toContain("text/html");
  await page.goto(base);
  await expect(page.locator("#start")).toBeEnabled({ timeout: 60_000 });
  await page.selectOption("#robotModel", "g1");
  await expect(page.locator("#robotModel")).toBeEnabled({ timeout: 60_000 });
  await expect(page.locator("#policy")).toHaveValue("g1Amp");
  await expect(page.locator("#engineState")).toHaveText("浏览器物理引擎已就绪");
  // A server fallback must be reported as a resource error, not broken MJCF.
  await page.reload();
  await expect(page.locator("#start")).toBeEnabled({ timeout: 60_000 });
  await page.route("**/robot/g1/scene.xml", (route) => route.fulfill({
    contentType: "text/html", body: "<!doctype html><html><body>Fallback page</body></html>",
  }));
  await page.selectOption("#robotModel", "g1");
  await expect(page.locator("#robotModel")).toBeEnabled({ timeout: 60_000 });
  await expect(page.locator("#engineState")).toContainText("返回了网页而非文件");
  await expect(page.locator("#robotModel")).toHaveValue("go2");
});
