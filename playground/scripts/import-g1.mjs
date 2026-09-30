// One-time, offline import. Normal builds use only files inside this repository.
import { readFile, writeFile, mkdir, copyFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const source = process.argv[2];
if (!source) throw new Error("Usage: node scripts/import-g1.mjs /path/to/wbc_fsm-main");
const assets = path.join(root, "playground/assets/g1");
const publicDir = path.join(root, "playground/public");
await mkdir(path.join(assets, "meshes"), { recursive: true });
await mkdir(path.join(publicDir, "policies"), { recursive: true });
const xml = await readFile(path.join(source, "assets/g1_description/g1_29dof_mjamp.xml"), "utf8");
await writeFile(path.join(assets, "scene.xml"), xml);
for (const [, file] of xml.matchAll(/<mesh\b[^>]*\bfile="([^"]+)"/g)) {
  const bytes = await readFile(path.join(source, "assets/g1_description", file));
  const shared = path.join(root, "src/assets/robots/unitree_g1/xmls/assets", path.basename(file));
  const existing = await readFile(shared).catch(() => null);
  if (!existing?.equals(bytes)) await writeFile(path.join(assets, file), bytes);
}
// Keep the original MJ AMP checkpoint byte-for-byte, published as G1 AMP.
await copyFile(path.join(source, "model/loco/Unitree-G1-AMP-Flat_model_30000.onnx"),
  path.join(publicDir, "policies/g1-amp.onnx"));
console.log("Imported G1 AMP policy and robot assets.");
