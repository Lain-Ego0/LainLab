import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "./tests",
  testMatch: "**/*.spec.mjs",
  timeout: 180_000,
  workers: 1,
  use: {
    baseURL: "http://127.0.0.1:4175",
    viewport: { width: 1440, height: 1000 },
    launchOptions: {
      executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH || undefined,
      args: ["--enable-unsafe-swiftshader"],
    },
  },
  webServer: [{
    command: "npm run build && npm run preview -- --host 127.0.0.1 --port 4175 --strictPort",
    url: "http://127.0.0.1:4175",
    timeout: 90_000,
  }, {
    command: "npm run dev -- --host 127.0.0.1 --port 4176 --strictPort",
    url: "http://127.0.0.1:4176",
    timeout: 90_000,
  }],
});
