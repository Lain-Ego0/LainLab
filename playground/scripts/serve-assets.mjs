import { readFile } from "node:fs/promises";
import path from "node:path";

// Read model assets from disk on each request, independently of Vite's watched
// public-file index. Missing model files must never fall back to index.html.
export function serveModelAssets(directory) {
  const root = path.resolve(directory);
  const mime = { ".xml": "application/xml", ".stl": "model/stl", ".obj": "text/plain", ".onnx": "application/octet-stream" };
  return async (request, response, next) => {
    let pathname;
    try { pathname = decodeURIComponent(request.url.split("?")[0]); }
    catch { response.writeHead(400).end("Invalid resource URL"); return; }
    if (!/^\/(robot|policies)\//.test(pathname)) { next(); return; }
    const file = path.resolve(root, `.${pathname}`);
    if (!file.startsWith(`${root}${path.sep}`)) { response.writeHead(403).end("Forbidden"); return; }
    if (request.method !== "GET" && request.method !== "HEAD") {
      response.writeHead(405, { Allow: "GET, HEAD" }).end(); return;
    }
    try {
      const bytes = await readFile(file);
      response.writeHead(200, {
        "Content-Type": mime[path.extname(file).toLowerCase()] || "application/octet-stream",
        "Content-Length": bytes.length,
        "Cache-Control": "no-cache",
      });
      response.end(request.method === "HEAD" ? undefined : bytes);
    } catch (error) {
      if (["ENOENT", "ENOTDIR", "EISDIR"].includes(error.code)) {
        response.writeHead(404, { "Content-Type": "text/plain; charset=utf-8", "Cache-Control": "no-store" }).end("Model resource not found");
      } else next(error);
    }
  };
}

export function modelAssetsPlugin() {
  return {
    name: "playground-model-assets",
    configureServer(server) {
      server.middlewares.use(serveModelAssets(server.config.publicDir));
    },
    configurePreviewServer(server) {
      server.middlewares.use(serveModelAssets(path.resolve(server.config.root, server.config.build.outDir)));
    },
  };
}
