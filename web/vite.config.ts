/// <reference types="vitest" />
import { createReadStream, existsSync, statSync } from "node:fs";
import { homedir } from "node:os";
import { join, normalize, resolve } from "node:path";
import { defineConfig, type Plugin } from "vite";

/** `npm run dev` serves the runs, scenes and catalogue from the checkout's data directories, as the
 *  console does: `showcase/`, then `runs/` and `data/`, then `$ROBOJEV_HOME`. */
function checkoutData(): Plugin {
  const repo = resolve(__dirname, "..");
  const home = process.env.ROBOJEV_HOME || join(homedir(), ".robojev");
  const roots: Record<string, string[]> = {
    replays: [join(repo, "showcase", "replays"), join(repo, "runs"), join(home, "replays")],
    scenes: [join(repo, "showcase", "scenes"), join(repo, "data", "scenes"), join(home, "scenes")],
    catalogue: [join(repo, "showcase", "catalogue"), join(repo, "data", "catalogue"), join(home, "catalogue")],
  };
  return {
    name: "robojev-checkout-data",
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        const path = decodeURIComponent((req.url ?? "").split("?")[0]).replace(/^\/+/, "");
        const [kind, ...rest] = path.split("/");
        if (!(kind in roots) || rest.some((p) => p === "..")) return next();
        for (const root of roots[kind]) {
          const file = normalize(join(root, ...rest));
          if (file.startsWith(root) && existsSync(file) && statSync(file).isFile()) {
            createReadStream(file).pipe(res);
            return;
          }
        }
        next();
      });
    },
  };
}
import react from "@vitejs/plugin-react";

/**
 * A static site, served as plain files from wherever it is dropped.
 *
 * `base: "./"` is the whole of the GitHub Pages story: every asset the build emits is referenced
 * relative to the page, so `https://<org>.github.io/RoboJEV/` and a local `dist/` opened under
 * some other sub-path are the same bundle. The MuJoCo worker is an ES module worker (robopp's
 * `new Worker(new URL(...), {type: "module"})`), so the worker bundle is built as one too.
 */
export default defineConfig({
  base: "./",
  plugins: [react(), checkoutData()],
  publicDir: false,
  // three.js is its own lazy chunk (the 3D stage), loaded after the page has painted; it is
  // bigger than Rollup's default warning and meant to be.
  // Runs, scenes and the catalogue are copied in by `scripts/publish-replays.mjs` after the build.
  build: { outDir: "dist", assetsInlineLimit: 0, chunkSizeWarningLimit: 700 },
  worker: { format: "es" },
  test: { environment: "node", include: ["src/**/*.test.ts"] },
});
