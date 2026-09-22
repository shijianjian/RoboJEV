"""Export a compiled MJCF plus every file it references into a self-contained, content-hashed bundle.

robopp's `robopp/scene.py`, ported: the same layout (`<scenes>/<hash>/scene.xml` and `assets/`),
the same optimisation and the same hash, so a scene exported here is byte for byte the bundle of
the same task in `data/scenes/` (robopp's own export, shipped with this repository). The web
viewer loads a bundle into MuJoCo WASM and sets qpos from the record. It never steps.

Bundles are also *optimised* on the way out (see `optimise_bundle`): text `.obj`/`.stl` meshes
become legacy binary `.msh`, and oversized PNG textures are capped. Both exist for one reason —
MuJoCo WASM re-parses every asset in the browser each time a scene is opened, and on a LIBERO
bundle that parse is what makes opening a task take tens of seconds rather than one.
"""
from __future__ import annotations

import dataclasses
import hashlib
import io
import logging
import os
import pathlib
import re
import shutil
import uuid
import xml.etree.ElementTree as ET

log = logging.getLogger(__name__)

MESHDIR_TAGS = {"mesh", "skin", "hfield"}
TEXTUREDIR_TAGS = {"texture"}

#: Longest side a bundled PNG texture may keep. 4096² textures are ~26 MB of PNG that the
#: browser has to decode before the first frame; 2048 halves each side without a visible
#: difference at the scales this viewer renders at.
DEFAULT_TEXTURE_MAX = 1024

#: Mesh formats MuJoCo parses as *text* — the ones worth rewriting as binary `.msh`.
MESH_SOURCE_SUFFIXES = (".obj", ".stl")

#: The marker `optimise_bundle` leaves in `scene.xml`, so a bundle says what was done to it. Kept
#: word for word: it is part of the bytes the hash covers, and a changed marker would give every
#: scene a hash different from the copy of the same scene in `data/scenes/`.
_TEXTURE_COMMENT = " robopp: textures capped at {max} "
_TEXTURE_COMMENT_RE = re.compile(r"^\s*robopp: textures capped at\b")

def optimise_enabled() -> bool:
    """Whether `export_scene_bundle` optimises what it writes. On unless `ROBOPP_BUNDLE_OPTIMISE=0`."""
    return (os.environ.get("ROBOJEV_BUNDLE_OPTIMISE") or os.environ.get("ROBOPP_BUNDLE_OPTIMISE", "1")) != "0"


def texture_max_from_env(default: int = DEFAULT_TEXTURE_MAX) -> int:
    """`ROBOPP_TEXTURE_MAX`, or `default` when unset or not a positive integer."""
    raw = os.environ.get("ROBOJEV_TEXTURE_MAX") or os.environ.get("ROBOPP_TEXTURE_MAX")
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        log.warning("ROBOPP_TEXTURE_MAX=%r is not an integer; using %d", raw, default)
        return default
    if value <= 0:
        log.warning("ROBOPP_TEXTURE_MAX=%d is not positive; using %d", value, default)
        return default
    return value


@dataclasses.dataclass
class OptimiseReport:
    """What one `optimise_bundle` call did, for logs, manifests and tests."""

    texture_max: int
    meshes_converted: int = 0
    #: How many of `meshes_converted` had to be de-indexed first (per-face-corner UVs).
    meshes_deindexed: int = 0
    #: How many of `meshes_converted` were byte-identical to a `.msh` already written for this
    #: bundle and so reference that file instead of a copy of their own (`_convert_meshes`).
    meshes_shared: int = 0
    textures_capped: int = 0
    original_bytes: int = 0
    bytes: int = 0
    #: `(mesh name, why)` for every `<mesh>` left in its original format.
    meshes_skipped: list[tuple[str, str]] = dataclasses.field(default_factory=list)


def bundle_bytes(bundle: pathlib.Path) -> int:
    return sum(p.stat().st_size for p in pathlib.Path(bundle).rglob("*") if p.is_file())


def write_replace(path: pathlib.Path, data: bytes) -> None:
    """Put `data` at `path` without ever truncating what is there.

    **The invariant this exists for**: after `robopp optimise-scenes --dedup`, a file under
    `<storage>/scenes/<hash>/` is very often a hardlink shared with dozens of other bundles (the
    robot meshes and textures every bundle re-ships — 47 links each, in this project's storage).
    Opening one for writing rewrites *every* bundle that shares the inode. So nothing in this
    module ever writes a bundle file in place: it writes a new file beside it and renames over the
    name, which leaves other links pointing at the old inode, untouched.

    The rename is atomic on POSIX, so a reader either sees the old file or the new one.
    """
    path = pathlib.Path(path)
    tmp = path.with_name(f".write-{os.getpid()}-{uuid.uuid4().hex[:8]}-{path.name}")
    try:
        tmp.write_bytes(data)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def bundle_hash(files: dict[str, bytes]) -> str:
    h = hashlib.sha256()
    for name in sorted(files):
        h.update(name.encode("utf-8") + b"\0")
        h.update(hashlib.sha256(files[name]).digest())
    return h.hexdigest()


def _resolve(file_attr: str, tag: str, compiler: ET.Element | None) -> pathlib.Path:
    p = pathlib.Path(file_attr)
    if p.is_absolute() or compiler is None:
        return p
    key = "meshdir" if tag in MESHDIR_TAGS else "texturedir" if tag in TEXTUREDIR_TAGS else None
    base = (compiler.get(key) if key else None) or compiler.get("assetdir")
    return pathlib.Path(base) / p if base else p


def _asset_name(stem: str, suffix: str, taken: set[str]) -> str:
    """A bundle-relative `assets/<name>` the viewer's own path check accepts.

    The viewer only fetches asset names matching `[A-Za-z0-9._-]+`, and a MuJoCo mesh name may
    carry anything, so anything else becomes `_`; a collision (two mesh names differing only in
    the replaced characters, or an asset already using the name) gets a numeric suffix.
    """
    base = re.sub(r"[^A-Za-z0-9._-]", "_", stem) or "mesh"
    name = f"{base}{suffix}"
    n = 1
    while name in taken:
        n += 1
        name = f"{base}_{n}{suffix}"
    taken.add(name)
    return name


def _legacy_msh(vert, normal, texcoord, face) -> bytes:
    """MuJoCo's legacy binary mesh: four int32 counts, then float32 vertices/normals/texcoords
    and int32 faces. The compiler reads it straight into its arrays — no text parsing, which is
    the whole point of writing it."""
    import numpy as np

    header = np.array([len(vert), len(normal), len(texcoord), len(face)], np.int32)
    return (
        header.tobytes()
        + np.ascontiguousarray(vert, np.float32).tobytes()
        + np.ascontiguousarray(normal, np.float32).tobytes()
        + np.ascontiguousarray(texcoord, np.float32).tobytes()
        + np.ascontiguousarray(face, np.int32).tobytes()
    )


def _convert_meshes(root: ET.Element, bundle: pathlib.Path, model, report: OptimiseReport) -> None:
    """Rewrite every text `<mesh file=...>` in `root` as a binary `.msh` beside it, writing each
    distinct payload once (see `by_payload` below).

    The vertex data comes from `model`, which has already parsed the originals, so this costs no
    second parse. MuJoCo stores a mesh's vertices in the frame it aligned them to, with
    `mesh_pos`/`mesh_quat` recording the alignment it applied and the referencing geom's
    `pos`/`quat` carrying the same transform; writing those aligned vertices out verbatim would
    therefore *drop* the mesh's own offset from the scene (the geom would pick the alignment up a
    second time from a mesh that no longer needs it, and e.g. LIBERO's pedestal moves half a
    metre). So the alignment is undone first: what lands in the `.msh` is the asset's own frame,
    which MuJoCo then re-aligns exactly as it did the original. `scale`/`refpos`/`refquat` are
    already baked into those vertices, hence dropped from the rewritten element.

    RoboCasa's compiled XML carries `content_type="model/obj"` on every `<mesh>` (237 of them in an
    atomic kitchen) because its meshes come from `.obj` files. MuJoCo dispatches its mesh decoder
    on that attribute before it looks at the extension, so a `file=` rewritten to `.msh` with the
    attribute left behind fails the whole model with `no decoder found for mesh file
    '…_collision_mesh_0.msh'` — on WASM 3.12 and on desktop 3.3.1 alike. LIBERO never hit it: its
    compiled XML emits no `content_type`.
    """
    import mujoco
    import numpy as np

    index = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, i): i for i in range(model.nmesh)}
    taken = {p.name for p in (bundle / "assets").iterdir()}
    converted: dict[str, str] = {}
    #: sha256 of a written `.msh` payload -> the asset name it was written under.
    by_payload: dict[str, str] = {}
    for elem in root.iter("mesh"):
        file_attr = elem.get("file")
        if not file_attr or not file_attr.lower().endswith(MESH_SOURCE_SUFFIXES):
            continue
        name = elem.get("name") or pathlib.PurePath(file_attr).stem
        i = index.get(name)
        if i is None:
            report.meshes_skipped.append((name, "not in the compiled model"))
            continue
        va, vn = int(model.mesh_vertadr[i]), int(model.mesh_vertnum[i])
        fa, fn = int(model.mesh_faceadr[i]), int(model.mesh_facenum[i])
        na, nn = int(model.mesh_normaladr[i]), int(model.mesh_normalnum[i])
        ta, tn = int(model.mesh_texcoordadr[i]), int(model.mesh_texcoordnum[i])
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, model.mesh_quat[i])
        rot = rot.reshape(3, 3)
        vert_src = model.mesh_vert[va:va + vn] @ rot.T + model.mesh_pos[i]
        face = model.mesh_face[fa:fa + fn]
        facetexcoord = model.mesh_facetexcoord[fa:fa + fn] if (ta >= 0 and tn) else None
        facenormal = model.mesh_facenormal[fa:fa + fn] if nn else None
        deindex = facetexcoord is not None and (tn != vn or not np.array_equal(facetexcoord, face))
        if deindex:
            # The legacy .msh format carries exactly one texcoord per *vertex*, but MuJoCo
            # indexes texcoords through `mesh_facetexcoord`, a per-face-corner array that need
            # not equal `mesh_face`: one vertex can carry different UVs on different faces
            # (36-42 % of a RoboCasa kitchen's meshes do). Writing vertex-ordered UVs would
            # come back with the texture permuted and pass every geometric check, so the mesh
            # used to be left as text `.obj`. De-indexing gives every face corner its own
            # vertex, which is exactly what the viewer's own buildThreeScene already does for
            # the non-indexed path: the UVs are preserved and the vertex count triples for those
            # meshes. That costs bundle size rather than saving it — measured on the spike's
            # atomic kitchen at cap 512, 60.7 MB with every mesh binary (356 converted, 127 of
            # them here) against 46.5 MB with those 127 left as text `.obj`. (The spike's 31.4 MB
            # figure for "all meshes binary" came from a per-file conversion that wrote
            # vertex-ordered UVs — the wrong ones — so it never paid for the extra vertices.)
            # What it buys is a bundle the browser can load at all without parsing text OBJ.
            corners = face.reshape(-1)
            vert_out = vert_src[corners]
            texcoord_out = model.mesh_texcoord[ta:ta + tn][facetexcoord.reshape(-1)]
            normal_out = ((model.mesh_normal[na:na + nn] @ rot.T)[facenormal.reshape(-1)]
                          if facenormal is not None else np.zeros((0, 3)))
            face_out = np.arange(fn * 3, dtype=np.int32).reshape(fn, 3)
            report.meshes_deindexed += 1
        else:
            vert_out = vert_src
            normal_out = model.mesh_normal[na:na + nn] @ rot.T if nn == vn else np.zeros((0, 3))
            texcoord_out = model.mesh_texcoord[ta:ta + tn] if (ta >= 0 and tn == vn) else np.zeros((0, 2))
            face_out = face

        payload = _legacy_msh(vert_out, normal_out, texcoord_out, face_out)
        # Content-addressed within the bundle: a kitchen instances the same mesh many times (three
        # identical stools, two identical light switches), and 348 `.msh` writes for the atomic
        # kitchen hold only 257 distinct payloads. Writing each distinct one once and pointing
        # every `<mesh>` that shares it at that file drops both the bytes and the fetches; the
        # `<mesh>` elements themselves are left alone, so the compiled model is unchanged --
        # `scale`/`refpos`/`refquat` are already baked into the vertices above, which is what
        # makes two elements with the same payload genuinely interchangeable.
        #
        # A bundle with no duplicates is byte-for-byte what it was before this: the first
        # occurrence of every payload takes the same `_asset_name` in the same order.
        digest = hashlib.sha256(payload).hexdigest()
        shared = by_payload.get(digest)
        if shared is None:
            out_name = _asset_name(name, ".msh", taken)
            # A brand-new name, so never a hardlink shared with another bundle - but written the
            # same way as everything else here, see write_replace.
            write_replace(bundle / "assets" / out_name, payload)
            by_payload[digest] = out_name
        else:
            out_name = shared
            report.meshes_shared += 1
        elem.set("file", f"assets/{out_name}")
        for attr in ("scale", "refpos", "refquat", "content_type"):
            elem.attrib.pop(attr, None)
        converted[name] = file_attr
        report.meshes_converted += 1

    if not converted:
        return
    # One file can back several `<mesh>` elements (the same .obj at two `scale`s), and a mesh
    # this pass skipped may still point at it, so a replaced original is only deleted once
    # nothing in the rewritten XML names it.
    still_used = {e.get("file") for e in root.iter() if e.get("file")}
    for old in set(converted.values()) - still_used:
        (bundle / old).unlink(missing_ok=True)


#: Pillow modes this pass refuses to touch: resampling them means going through 8-bit RGB, which
#: silently destroys the data rather than merely shrinking it.
_UNSUPPORTED_TEXTURE_MODES = {"I", "I;16", "I;16B", "I;16L", "I;16N", "F"}


def _cap_textures(root: ET.Element, bundle: pathlib.Path, texture_max: int, report: OptimiseReport) -> None:
    """Downscale every PNG texture whose longest side exceeds `texture_max`.

    Cube and skybox textures are laid out as a grid of faces in one image, so they are halved
    (an exact integer division of every face) rather than fitted to an arbitrary side length.

    The image's own kind is kept: a palette PNG is re-quantised back to a palette (going to
    24-bit RGB can leave a *downscaled* image larger than the original), greyscale stays
    greyscale, and a high-bit-depth image is left alone entirely rather than clamped to 8 bits.
    The result is written beside the original and renamed over it, never into it — see
    `write_replace`.
    """
    from PIL import Image

    types: dict[str, set[str]] = {}
    for elem in root.iter("texture"):
        file_attr = elem.get("file")
        if file_attr and file_attr.lower().endswith(".png"):
            types.setdefault(file_attr, set()).add(elem.get("type") or "cube")
    for file_attr, kinds in sorted(types.items()):
        path = bundle / file_attr
        try:
            with Image.open(path) as im:
                width, height = im.size
                if max(width, height) <= texture_max:
                    continue
                if im.mode in _UNSUPPORTED_TEXTURE_MODES:
                    log.warning("texture %s is mode %s; left at %dx%d", file_attr, im.mode, width, height)
                    continue
                target = _texture_size(width, height, texture_max, grid=bool(kinds & {"cube", "skybox"}))
                data = _resized(im, target)
            buf = io.BytesIO()
            data.save(buf, format="PNG", optimize=True)
            write_replace(path, buf.getvalue())
            report.textures_capped += 1
        except OSError as e:
            log.warning("texture %s could not be capped: %s", file_attr, e)


def _resized(im, target: tuple[int, int]):
    """LANCZOS-resample `im` to `target`, coming back out in the kind of image it went in as."""
    from PIL import Image

    alpha = im.mode in ("RGBA", "LA", "PA") or "transparency" in im.info
    if im.mode == "P":
        # A palette cannot be resampled directly (LANCZOS would interpolate palette *indices*),
        # so it goes through full colour and is quantised back.
        through = "RGBA" if alpha else "RGB"
        small = im.convert(through).resize(target, Image.LANCZOS)
        return small.quantize(colors=min(len(im.getpalette() or []) // 3 or 256, 256))
    if im.mode in ("L", "LA"):
        return im.resize(target, Image.LANCZOS)
    return im.convert("RGBA" if alpha else "RGB").resize(target, Image.LANCZOS)


def _texture_size(width: int, height: int, texture_max: int, grid: bool) -> tuple[int, int]:
    if grid:
        factor = 1
        while max(width, height) // factor > texture_max and width // (factor * 2) and height // (factor * 2):
            factor *= 2
        return max(width // factor, 1), max(height // factor, 1)
    scale = texture_max / max(width, height)
    return max(round(width * scale), 1), max(round(height * scale), 1)


def _parse_keeping_comments(xml_path: pathlib.Path) -> ET.Element:
    """`ET.parse` drops comments, which would quietly delete any a bundle's XML carried (and make
    this module's own marker survive only because it is re-added). Keeping them means the rewrite
    changes what it means to change, and the marker's de-duplication below is real."""
    try:
        parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
        parser.feed(xml_path.read_bytes())
        return parser.close()
    except ET.ParseError:
        # A comment outside the root element is legal XML that a comment-keeping TreeBuilder has
        # nowhere to put. Fall back to the comment-dropping parse rather than refuse the bundle.
        log.warning("%s: could not be parsed with comments; they will not survive the rewrite", xml_path)
        return ET.parse(xml_path).getroot()


def optimise_bundle(bundle: pathlib.Path, texture_max: int = DEFAULT_TEXTURE_MAX) -> OptimiseReport:
    """Rewrite one already-materialised bundle directory into its browser-friendly form.

    Idempotent: a second call finds no text meshes and no oversized textures left, and rewrites
    only the marker comment. Raises whatever MuJoCo raises if `scene.xml` does not compile —
    callers decide whether that is fatal.

    Every file it writes is written beside its target and renamed over it (`write_replace`), so a
    bundle whose files are hardlinked into other bundles by `optimise-scenes --dedup` is still
    safe to rewrite: the other bundles keep the inode they had.
    """
    import mujoco

    bundle = pathlib.Path(bundle)
    xml_path = bundle / "scene.xml"
    report = OptimiseReport(texture_max=texture_max, original_bytes=bundle_bytes(bundle))
    root = _parse_keeping_comments(xml_path)
    if next(root.iter("include"), None) is not None:
        # `ET` does not expand <include>, so a mesh declared in an included file would be invisible
        # here while present in the compiled model - and the "is this original still referenced"
        # scan would not see the include's references either, so it could delete a file the
        # include still names. `sim.model.get_xml()` is always flat, so this never fires; it is
        # here so it fails loudly rather than mangling the bundle if that ever changes.
        raise ValueError(f"{xml_path}: <include> is not supported by the bundle optimiser")
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    try:
        _convert_meshes(root, bundle, model, report)
        _cap_textures(root, bundle, texture_max, report)
    finally:
        del model
    for child in [c for c in root if c.tag is ET.Comment and _TEXTURE_COMMENT_RE.match(c.text or "")]:
        root.remove(child)
    root.insert(0, ET.Comment(_TEXTURE_COMMENT.format(max=texture_max)))
    write_replace(xml_path, ET.tostring(root, encoding="unicode").encode("utf-8"))
    report.bytes = bundle_bytes(bundle)
    return report


def export_scene_bundle(
    xml: str,
    scenes_dir: pathlib.Path,
    optimise: bool | None = None,
    texture_max: int | None = None,
) -> str:
    """Write `xml` and everything it references into `<scenes_dir>/<hash>/`, returning the hash.

    `texture_max` is the caller's cap for this bundle's PNG textures, and comes from the suite:
    `registry.suites.<id>.scene.texture_max` (512 for RoboCasa, measured strictly better there —
    ~10 MB smaller per kitchen and ~400 MB less WASM heap; `DEFAULT_TEXTURE_MAX`, 1024, everywhere
    else). `ROBOPP_TEXTURE_MAX` still overrides whatever the caller passes, and `None` means
    `DEFAULT_TEXTURE_MAX`, so a caller that says nothing keeps today's behaviour. `robopp
    .scenes_optimise` deliberately keeps its own explicit `--texture-max` instead of this: it
    rewrites bundles on disk whose suite it no longer knows.

    """
    root = ET.fromstring(xml)
    compiler = root.find("compiler")

    files: dict[str, bytes] = {}
    chosen: dict[pathlib.Path, str] = {}
    for elem in root.iter():
        f = elem.get("file")
        if not f:
            continue
        src = _resolve(f, elem.tag, compiler).resolve()
        if src not in chosen:
            try:
                data = src.read_bytes()
            except OSError as e:
                raise FileNotFoundError(f"<{elem.tag} file={f!r}> could not be resolved") from e
            name = src.name
            if f"assets/{name}" in files:
                name = hashlib.sha256(data).hexdigest()[:8] + "_" + name
            chosen[src] = f"assets/{name}"
            files[chosen[src]] = data
        elem.set("file", chosen[src])

    if compiler is not None:
        for attr in ("meshdir", "texturedir"):
            compiler.attrib.pop(attr, None)

    files["scene.xml"] = ET.tostring(root, encoding="unicode").encode("utf-8")

    # Stage under a per-call-unique name. The staged copy is what gets optimised (the pass needs
    # the files on disk to compile them), so unlike before the hash is only known once staging is
    # done — it has to cover the optimised bytes, not the originals. A leftover ".partial-*"
    # directory from a crashed run is simply orphaned and can be ignored (or cleaned up
    # out-of-band); we never reuse or delete it here.
    scenes_dir = pathlib.Path(scenes_dir)
    tmp = scenes_dir / f".partial-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    (tmp / "assets").mkdir(parents=True)
    for name, data in files.items():
        (tmp / name).write_bytes(data)
    if optimise_enabled() if optimise is None else optimise:
        try:
            report = optimise_bundle(
                tmp, texture_max_from_env(DEFAULT_TEXTURE_MAX if texture_max is None else texture_max)
            )
            log.info(
                "scene bundle optimised: %d meshes (%d de-indexed), %d textures, %d -> %d bytes",
                report.meshes_converted, report.meshes_deindexed, report.textures_capped,
                report.original_bytes, report.bytes,
            )
        except Exception:
            # An export that produced a valid bundle must not fail because it could not be made
            # smaller; the unoptimised bundle is still correct, just slower to open.
            log.exception("scene bundle optimisation failed; exporting the bundle unoptimised")
            shutil.rmtree(tmp, ignore_errors=True)
            tmp = scenes_dir / f".partial-{os.getpid()}-{uuid.uuid4().hex[:8]}"
            (tmp / "assets").mkdir(parents=True)
            for name, data in files.items():
                (tmp / name).write_bytes(data)

    files = {
        str(p.relative_to(tmp)).replace(os.sep, "/"): p.read_bytes()
        for p in sorted(tmp.rglob("*")) if p.is_file()
    }
    h = bundle_hash(files)

    out = scenes_dir / h
    if out.exists():
        shutil.rmtree(tmp, ignore_errors=True)
        return h
    try:
        tmp.rename(out)
    except OSError:
        # Another concurrent caller won the race and already created `out`.
        # Treat that as success and discard our staging directory.
        shutil.rmtree(tmp, ignore_errors=True)
        if not out.exists():
            raise
    return h


# ----------------------------------------------------------------------------------- where scenes are

#: The checkout's three data directories: `showcase/` (tracked: the runs GitHub carries, and the
#: scenes and catalogue entries they use), `data/` (not tracked: the full catalogue and compiled
#: scenes) and `runs/` (not tracked: what the console saves and `robojev record` writes).
CHECKOUT = pathlib.Path(__file__).resolve().parents[1]
SHOWCASE = CHECKOUT / "showcase"
REPO_DATA = CHECKOUT / "data"
RUNS = CHECKOUT / "runs"


def scene_roots() -> list[pathlib.Path]:
    """Where compiled scenes are looked for, in order: `showcase/scenes`, `data/scenes`, then
    `$ROBOJEV_HOME/scenes` (what this machine exported or fetched)."""
    from robojev.home import home

    return [SHOWCASE / "scenes", REPO_DATA / "scenes", home() / "scenes"]


def locate(scene_hash: str) -> pathlib.Path | None:
    """The directory holding `scene_hash`, or None."""
    for root in scene_roots():
        if (root / scene_hash / "scene.xml").is_file():
            return root / scene_hash
    return None


def ensure_scene(xml: str, texture_max: int | None = None) -> str:
    """The hash of `xml`'s bundle, exported into `$ROBOJEV_HOME/scenes` unless a root already
    holds it. The export is the only way to learn the hash, so it is always run; a bundle the
    repository already has is then dropped from the home directory rather than kept twice."""
    from robojev.home import home

    home_scenes = home() / "scenes"
    home_scenes.mkdir(parents=True, exist_ok=True)
    had = (home_scenes).exists() and {p.name for p in home_scenes.iterdir()}
    digest = export_scene_bundle(xml, home_scenes, texture_max=texture_max)
    in_checkout = any((r / digest / "scene.xml").is_file() for r in scene_roots()[:2])
    if in_checkout and digest not in (had or set()):
        shutil.rmtree(home_scenes / digest, ignore_errors=True)
    return digest
