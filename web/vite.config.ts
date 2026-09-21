/// <reference types="vitest" />
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

/**
 * A static site, served as plain files from wherever it is dropped.
 *
 * `base: "./"` is the whole of the GitHub Pages story: every asset the build emits is referenced
 * relative to the page, so `https://<org>.github.io/RoboJEV/` and a local `dist/` opened under
 * some other sub-path are the same bundle. Nothing here talks to a server, so there is no proxy
 * and no API base to configure -- the one thing a future live console would need is
 * `VITE_LIVE_URL`, read at runtime in `src/data/source.ts` and documented in `PROTOCOL.md`.
 */
export default defineConfig({
  base: "./",
  plugins: [react()],
  build: { outDir: "dist", assetsInlineLimit: 0 },
  test: { environment: "node", include: ["src/**/*.test.ts"] },
});
