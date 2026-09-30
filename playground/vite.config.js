import { defineConfig } from "vite";
import { modelAssetsPlugin } from "./scripts/serve-assets.mjs";

// The playground is published below /playground/ on the personal website.
// Relative output keeps it portable if the parent site's URL changes.
export default defineConfig({ base: "./", plugins: [modelAssetsPlugin()] });
