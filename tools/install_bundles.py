"""Copy recorded bundles into the web app and write `replays/index.json`.

The index is derived, never hand-written: a strip item is a projection of the bundle beside it,
so the one way for the two to disagree is for a human to type the second one. The poster is
derived too -- a bundle that shipped before posters existed gets one here, out of its own video.

    python3 tools/install_bundles.py work/bowl-plate work/drawer work/cookie-box work/failure
"""
from __future__ import annotations

import json
import pathlib
import shutil
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from robojev.recorder import POSTER_NAME, poster_time, write_poster   # noqa: E402

REPLAYS = ROOT / "web" / "public" / "replays"

#: The order the episode strip lists them in, and so which one is first. A bundle not named here
#: goes to the end, alphabetically.
ORDER = ["drawer", "bowl-plate", "cookie-box", "failure"]


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    REPLAYS.mkdir(parents=True, exist_ok=True)
    entries = []
    for arg in argv:
        src = pathlib.Path(arg).resolve()
        bundle = json.loads((src / "episode.json").read_text(encoding="utf-8"))
        dst = REPLAYS / bundle["id"]
        if dst.exists():
            shutil.rmtree(dst)
        dst.mkdir(parents=True)
        for name in ("episode.json", "agentview.mp4", "wrist.mp4", POSTER_NAME):
            if (src / name).exists():
                shutil.copy2(src / name, dst / name)
        # Derived here as well as at record time, so the strip has a picture for every bundle
        # whatever version of the recorder wrote it.
        poster = bundle.get("poster") or write_poster(dst, poster_time(bundle))
        if poster is not None and not (dst / poster).exists():
            poster = write_poster(dst, poster_time(bundle))
        if bundle.get("poster") != poster:
            bundle["poster"] = poster
            (dst / "episode.json").write_text(json.dumps(bundle, indent=1) + "\n", encoding="utf-8")
        entries.append({
            "id": bundle["id"],
            "title": bundle["title"],
            "instruction": bundle["instruction"],
            "note": bundle["note"],
            "success": bundle["success"],
            "decisions": len(bundle["decisions"]),
            "max_decisions": bundle["max_decisions"],
            "task_index": bundle["task_index"],
            "init_state_index": bundle["init_state_index"],
            "suite": bundle["suite"],
            "poster": poster,
        })
        size = sum(f.stat().st_size for f in dst.iterdir())
        print(f"{bundle['id']:<12} success={bundle['success']!s:<5} "
              f"decisions={len(bundle['decisions']):>3}/{bundle['max_decisions']} "
              f"{size / 1e6:.2f} MB")

    entries.sort(key=lambda e: (ORDER.index(e["id"]) if e["id"] in ORDER else len(ORDER), e["id"]))
    (REPLAYS / "index.json").write_text(json.dumps(entries, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {REPLAYS / 'index.json'} ({len(entries)} bundles)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
