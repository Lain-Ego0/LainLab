import { mkdir, readFile, writeFile, readdir, access } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import path from "node:path";

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, "../..");
const source = path.join(root, "src/assets/robots/unitree_go2/xmls");
const target = path.join(root, "playground/public/robot");

async function syncFile(from, to) {
  const bytes = await readFile(from);
  const previous = await readFile(to).catch((error) => {
    if (error.code !== "ENOENT") throw error;
    return null;
  });
  if (previous?.equals(bytes)) return;
  await mkdir(path.dirname(to), { recursive: true });
  // fs.cp unlinks existing files before copying, even if the directory remains.
  // Preserve the inode so a running Vite watcher never receives that unlink.
  await writeFile(to, bytes);
}

async function syncDirectory(from, to) {
  for (const entry of await readdir(from, { withFileTypes: true })) {
    const input = path.join(from, entry.name), output = path.join(to, entry.name);
    if (entry.isDirectory()) await syncDirectory(input, output);
    else if (entry.isFile()) await syncFile(input, output);
  }
}

await syncDirectory(source, target);

const g1 = path.join(root, "playground/assets/g1");
const g1Target = path.join(target, "g1");
await mkdir(path.join(g1Target, "meshes"), { recursive: true });
await syncFile(path.join(g1, "scene.xml"), path.join(g1Target, "scene.xml"));
const xml = await readFile(path.join(g1, "scene.xml"), "utf8");
for (const [, file] of xml.matchAll(/<mesh\b[^>]*\bfile="([^"]+)"/g)) {
  const override = path.join(g1, file);
  const exists = await access(override).then(() => true, () => false);
  const shared = path.join(root, "src/assets/robots/unitree_g1/xmls/assets", path.basename(file));
  await syncFile(exists ? override : shared, path.join(g1Target, file));
}
