/**
 * A plain static file server for `web/dist`, mounted under a sub-path — what GitHub Pages does,
 * with none of what it does extra. It exists to prove the built site is *only* files: no rewrite
 * rules, no SPA fallback, no API.
 *
 * It does implement byte ranges, because a browser seeking an mp4 asks for one and a server that
 * answers 200-with-the-whole-file to a Range request makes seeking silently unreliable — which
 * would be the site's own bug hiding behind the harness's.
 *
 *     node tools/serve.mjs [port] [mount] [root]   # default: 8137 /RoboJEV/ web/dist
 */
import { createReadStream, statSync } from "node:fs";
import { createServer } from "node:http";
import { extname, join, normalize, resolve } from "node:path";

const PORT = Number(process.argv[2] ?? 8137);
const MOUNT = (process.argv[3] ?? "/RoboJEV/").replace(/\/*$/, "/");
const ROOT = resolve(process.argv[4] ??
  resolve(new URL(".", import.meta.url).pathname, "..", "web", "dist"));

const TYPES = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".mp4": "video/mp4",
  ".svg": "image/svg+xml",
  ".png": "image/png",
  ".ico": "image/x-icon",
};

createServer((req, res) => {
  const url = new URL(req.url ?? "/", "http://localhost");
  let path = decodeURIComponent(url.pathname);
  if (!path.startsWith(MOUNT)) {
    res.writeHead(404, { "content-type": "text/plain" });
    res.end(`not under ${MOUNT}\n`);
    return;
  }
  path = path.slice(MOUNT.length) || "index.html";
  if (path.endsWith("/")) path += "index.html";
  const file = join(ROOT, normalize(path).replace(/^(\.\.[/\\])+/, ""));

  let info;
  try {
    info = statSync(file);
    if (info.isDirectory()) throw new Error("directory");
  } catch {
    res.writeHead(404, { "content-type": "text/plain" });
    res.end("not found\n");
    return;
  }

  const type = TYPES[extname(file)] ?? "application/octet-stream";
  const range = /^bytes=(\d*)-(\d*)$/.exec(req.headers.range ?? "");
  if (range !== null) {
    const start = range[1] === "" ? Math.max(0, info.size - Number(range[2])) : Number(range[1]);
    const end = range[2] === "" || range[1] === "" ? info.size - 1 : Math.min(Number(range[2]), info.size - 1);
    if (start >= info.size || start > end) {
      res.writeHead(416, { "content-range": `bytes */${info.size}` });
      res.end();
      return;
    }
    res.writeHead(206, {
      "content-type": type,
      "content-length": end - start + 1,
      "content-range": `bytes ${start}-${end}/${info.size}`,
      "accept-ranges": "bytes",
    });
    createReadStream(file, { start, end }).pipe(res);
    return;
  }

  res.writeHead(200, { "content-type": type, "content-length": info.size, "accept-ranges": "bytes" });
  createReadStream(file).pipe(res);
}).listen(PORT, "127.0.0.1", () => {
  console.log(`serving ${ROOT} at http://127.0.0.1:${PORT}${MOUNT}`);
});
