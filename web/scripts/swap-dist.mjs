/** Put a finished build in place of `dist/` in one rename, so a console serving `dist/` from disk
 *  never serves a half-written page. The build is made in `.dist-next/` and swapped here. */
import { existsSync, renameSync, rmSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const WEB = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const next = join(WEB, ".dist-next"), dist = join(WEB, "dist"), old = join(WEB, ".dist-old");
rmSync(old, { recursive: true, force: true });
if (existsSync(dist)) renameSync(dist, old);
renameSync(next, dist);
rmSync(old, { recursive: true, force: true });
