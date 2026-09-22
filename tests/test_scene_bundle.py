"""The 3D scene a bundle carries: robopp's compiled scene export, and the pose table.

The optimising half of the exporter needs MuJoCo and Pillow; what is tested here is the half that
decides the layout and the hash - robopp's, so that a scene exported here is the copy of the same
scene the repository ships - and it runs with `optimise=False` on numpy alone.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from robojev import recorder, scene_bundle
from robojev.console.images import FrameBuffer
from robojev.console.session import Session

from test_console_session import RenderEnv, Sink, StubEnv, finish, send, start

XML = """<mujoco model="t">
  <compiler meshdir="{meshes}" texturedir="{textures}"/>
  <asset>
    <mesh name="bowl" file="bowl.stl"/>
    <texture name="wood" type="2d" file="wood.png"/>
  </asset>
  <worldbody><body name="b"><geom type="mesh" mesh="bowl"/></body></worldbody>
</mujoco>"""


def make_assets(tmp_path, bowl=b"solid bowl\n"):
    meshes, textures = tmp_path / "meshes", tmp_path / "textures"
    meshes.mkdir(parents=True, exist_ok=True)
    textures.mkdir(parents=True, exist_ok=True)
    (meshes / "bowl.stl").write_bytes(bowl)
    (textures / "wood.png").write_bytes(b"\x89PNG not really")
    return XML.format(meshes=meshes, textures=textures)


def test_a_scene_is_its_xml_and_its_assets_under_its_hash(tmp_path):
    scenes = tmp_path / "scenes"
    digest = scene_bundle.export_scene_bundle(make_assets(tmp_path), scenes, optimise=False)
    written = (scenes / digest / "scene.xml").read_text()
    assert sorted(p.name for p in (scenes / digest / "assets").iterdir()) == ["bowl.stl", "wood.png"]
    assert 'file="assets/bowl.stl"' in written and "meshdir" not in written
    files = {"scene.xml": written.encode(), "assets/bowl.stl": b"solid bowl\n", "assets/wood.png": b"\x89PNG not really"}
    assert digest == scene_bundle.bundle_hash(files)


def test_the_same_scene_exports_to_the_same_hash_and_different_bytes_to_another(tmp_path):
    scenes = tmp_path / "scenes"
    a = scene_bundle.export_scene_bundle(make_assets(tmp_path / "a", b"one"), scenes, optimise=False)
    again = scene_bundle.export_scene_bundle(make_assets(tmp_path / "b", b"one"), scenes, optimise=False)
    other = scene_bundle.export_scene_bundle(make_assets(tmp_path / "c", b"two"), scenes, optimise=False)
    assert a == again and a != other


def test_a_scene_is_found_in_the_checkout_before_the_home_directory(tmp_path, monkeypatch):
    show = tmp_path / "showcase"
    (show / "scenes" / "abc").mkdir(parents=True)
    (show / "scenes" / "abc" / "scene.xml").write_text("<mujoco/>")
    monkeypatch.setattr(scene_bundle, "SHOWCASE", show)
    monkeypatch.setattr(scene_bundle, "REPO_DATA", tmp_path / "data")
    monkeypatch.setenv("ROBOJEV_HOME", str(tmp_path / "home"))
    assert scene_bundle.locate("abc") == show / "scenes" / "abc"
    assert scene_bundle.locate("nope") is None


def test_the_pose_table_round_trips_as_little_endian_float32(tmp_path):
    rows = [np.arange(4, dtype=np.float64) + i for i in range(3)]
    spec = recorder.write_qpos(tmp_path, rows)
    assert spec == {"path": "qpos.bin", "frames": 3, "nq": 4, "dtype": "<f4"}
    assert (tmp_path / "qpos.bin").stat().st_size == 3 * 4 * 4
    back = recorder.read_qpos(tmp_path, {"qpos": spec})
    assert back.shape == (3, 4) and np.allclose(back, np.stack(rows))
    # A frame with no pose means no table at all, rather than a table with a hole in it.
    assert recorder.write_qpos(tmp_path / "none", [rows[0], None]) is None


# ------------------------------------------------------------------------- the live console

class SceneEnv(RenderEnv):
    """A stub with a pose and an MJCF, like the LIBERO adapter."""

    def __init__(self, *a, xml="<mujoco/>", **kw):
        super().__init__(*a, **kw)
        self._xml = xml

    def ground_truth(self):
        q = np.array([*self._eef, self._width], dtype=np.float32)
        return q, np.zeros(4, np.float32), float(self.steps)

    def scene_xml(self) -> str:
        return self._xml


def scene_session(tmp_path, env):
    from robojev import policy as policy_mod

    sink = Sink()
    session = Session(
        sink, frames=FrameBuffer(), replays_dir=tmp_path / "replays",
        base_url="http://127.0.0.1:8765", policies=("expert",), render_size=8,
        scenes_dir=tmp_path / "scenes",
        env_factory=lambda suite, task, render_size=256, env_seed=0: env,
        build_policy=lambda spec: policy_mod.build("expert", spec.suite, seed=spec.seed,
                                                   selection=spec.selection))
    return session, sink


@pytest.fixture
def no_optimise(monkeypatch):
    monkeypatch.setenv("ROBOJEV_BUNDLE_OPTIMISE", "0")


def test_a_live_episode_names_its_scene_and_streams_its_pose(tmp_path, no_optimise):
    env = SceneEnv(done_at=10, xml=make_assets(tmp_path))
    session, sink = scene_session(tmp_path, env)
    hello = start(session, sink)
    scene = hello["scene"]
    assert scene["nq"] == 4 and scene["xml"] == f"http://127.0.0.1:8765/scenes/{scene['hash']}/scene.xml"
    assert scene["assets"] == f"http://127.0.0.1:8765/scenes/{scene['hash']}/assets/"
    assert (tmp_path / "scenes" / scene["hash"] / "scene.xml").is_file()
    first = sink.wait("pose")
    assert len(first["qpos"]) == 4
    send(session, "step")
    sink.wait("decision")
    steps = [m["step"] for m in sink.of("pose")]
    assert steps == sorted(steps) and len(steps) > 5
    finish(session)


def test_a_saved_live_episode_carries_its_pose_table_and_names_its_scene(tmp_path, no_optimise):
    env = SceneEnv(done_at=10, xml=make_assets(tmp_path))
    session, sink = scene_session(tmp_path, env)
    hello = start(session, sink)
    send(session, "run")
    sink.wait("done")
    send(session, "save")
    saved = sink.wait("saved", timeout=30.0)
    out = tmp_path / "replays" / saved["id"]
    bundle = json.loads((out / "episode.json").read_text())
    assert bundle["scene"] == {"hash": hello["scene"]["hash"], "nq": 4}
    assert recorder.read_qpos(out, bundle).shape == (bundle["total_frames"], 4)
    finish(session)


def test_an_environment_with_no_scene_starts_without_one(tmp_path):
    session, sink = scene_session(tmp_path, StubEnv(done_at=10))
    assert start(session, sink)["scene"] is None
    finish(session)
