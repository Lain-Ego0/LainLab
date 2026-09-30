import test from "node:test";
import assert from "node:assert/strict";
import { createServer } from "node:http";
import { mkdtemp, mkdir, writeFile, rename, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { serveModelAssets } from "../scripts/serve-assets.mjs";

test("model serving follows disk updates and never falls back to HTML for missing meshes", async (t) => {
  const root = await mkdtemp(path.join(tmpdir(), "playground-assets-"));
  const middleware = serveModelAssets(root);
  const server = createServer((req, res) => middleware(req, res, () => {
    res.writeHead(200, { "Content-Type": "text/html" }).end("<html>app</html>");
  }));
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  t.after(async () => {
    await new Promise((resolve) => server.close(resolve));
    await rm(root, { recursive: true, force: true });
  });
  const base = `http://127.0.0.1:${server.address().port}`;
  const url = `${base}/robot/g1/meshes/right_elbow_link.STL`;
  const missing = await fetch(url);
  assert.equal(missing.status, 404);
  assert.doesNotMatch(missing.headers.get("content-type"), /html/);
  const file = path.join(root, "robot/g1/meshes/right_elbow_link.STL");
  await mkdir(path.dirname(file), { recursive: true });
  await writeFile(file, "first mesh");
  assert.equal(await (await fetch(url)).text(), "first mesh");
  await writeFile(`${file}.tmp`, "updated mesh");
  await rename(`${file}.tmp`, file);
  assert.equal(await (await fetch(url)).text(), "updated mesh");
  const head = await fetch(url, { method: "HEAD" });
  assert.equal(head.status, 200);
  assert.equal(head.headers.get("content-type"), "model/stl");
  assert.equal(head.headers.get("content-length"), "12");
  assert.equal(await head.text(), "");
  await rm(file);
  assert.equal((await fetch(url)).status, 404);
  assert.equal((await fetch(`${base}/robot/%2e%2e%2f%2e%2e%2foutside`)).status, 403);
  assert.equal(await (await fetch(base)).text(), "<html>app</html>");
});
