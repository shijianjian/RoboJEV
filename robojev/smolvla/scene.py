"""The XLeRobot at a table: its URDF model with the base parked, LIBERO's bowl, plate and ramekin,
and the robot's three cameras (head, left wrist, right wrist).

The table is pressed against the cart's front face and lowered to 0.74 m: the XLeRobot's arms reach
only 12-16 cm past the cart (measured on the URDF), so everything sits in that band. A layout is a
seed-drawn arrangement plus appearance (table colour, light, camera jitter).
"""
from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FRONT = 0.200  # x of the cart's front face
TABLE_Z = 0.74
ARM_JOINTS = ("Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll")
SUFFIX = {"right": "", "left": "_2"}  # the URDF: unsuffixed joints are the right arm, `_2` the left
IMAGE = 256  # every camera renders IMAGE x IMAGE, SmolVLA's input size
OBJECTS = {  # name -> (LIBERO folder, xml)
    "bowl": ("akita_black_bowl", "akita_black_bowl.xml"),
    "plate": ("plate", "plate.xml"),
    "ramekin": ("glazed_rim_porcelain_ramekin", "glazed_rim_porcelain_ramekin.xml"),
}


def libero_objects() -> Path:
    """LIBERO's `stable_scanned_objects` folder. Only the object files are used, not LIBERO's code
    (which pins an older Python than LeRobot allows): point ROBOJEV_LIBERO_OBJECTS at the folder."""
    p = os.environ.get("ROBOJEV_LIBERO_OBJECTS")
    if p:
        return Path(p)
    try:  # an installed LIBERO knows where its assets are
        from libero.libero import get_libero_path

        return Path(get_libero_path("assets")) / "stable_scanned_objects"
    except Exception as e:  # noqa: BLE001
        raise RuntimeError("set ROBOJEV_LIBERO_OBJECTS to LIBERO's assets/stable_scanned_objects folder") from e


@dataclass
class Layout:
    arm: str  # the arm that picks: the one on the bowl's side
    bowl: tuple[float, float]
    plate: tuple[float, float]
    ramekin: tuple[float, float] | None
    table_rgb: tuple[float, float, float] = (0.55, 0.40, 0.28)
    light: float = 0.9
    head_jitter: tuple[float, float, float] = (0.0, 0.0, 0.0)  # metres
    extra: dict = field(default_factory=dict)


def sample_layout(seed: int) -> Layout:
    """A layout in the reach band of one arm. Right-arm layouts are the measured spike layouts'
    ranges (bowl 6-10 cm past the cart, 12-22 cm right of centre; plate 6.5-8.5 cm past, 15-26 cm
    toward the midline from the bowl and at most 4 cm across it);
    left-arm layouts are their mirror image."""
    rng = np.random.default_rng(seed)
    arm = "right" if rng.random() < 0.5 else "left"
    s = -1.0 if arm == "right" else 1.0  # the bowl's side (right arm: -y)
    bowl = (FRONT + float(rng.uniform(0.06, 0.10)), s * float(rng.uniform(0.12, 0.22)))
    # the plate toward the midline from the bowl, far enough that they do not overlap (plate radius
    # ~10 cm + bowl 4.7 cm: closer and the bowl spawned inside the plate and was flung off the
    # table), and at most 4 cm across the midline (an arm reaching farther cannot lower the bowl
    # level; it ended tilted 30-40 deg, still pinched)
    gap = float(rng.uniform(0.15, min(0.26, abs(bowl[1]) + 0.04)))
    plate = (FRONT + float(rng.uniform(0.065, 0.085)), s * (abs(bowl[1]) - gap))
    ramekin = None
    if rng.random() < 0.5:  # a distractor on the far side of the bowl from the plate, when there is room
        y = bowl[1] + s * 0.12
        if abs(y) < 0.36:
            ramekin = (bowl[0] + float(rng.uniform(-0.01, 0.02)), y)
    table = tuple(float(v) for v in np.clip(np.array([0.55, 0.40, 0.28]) + rng.uniform(-0.15, 0.15, 3), 0.1, 0.9))
    return Layout(arm, bowl, plate, ramekin, table, float(rng.uniform(0.6, 1.1)),
                  tuple(float(v) for v in rng.uniform(-0.01, 0.01, 3)))


def _lookat_xyaxes(pos, target) -> str:
    f = np.asarray(target, float) - np.asarray(pos, float)
    f /= np.linalg.norm(f)
    z = -f
    x = np.array([-z[1], z[0], 0.0])  # horizontal: no roll
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return " ".join(f"{v:.5f}" for v in (*x, *y))


def _object(name: str, root: Path, pos) -> tuple[list, ET.Element]:
    folder, xml = OBJECTS[name]
    o = ET.parse(root / folder / xml).getroot()
    for a in o.iter():
        if a.get("file"):
            a.set("file", str(root / folder / a.get("file")))
    assets = list(o.find("asset"))
    for a in assets:
        if a.tag == "material":
            a.set("reflectance", "0")  # LIBERO's 0.5 mirrors the table: its black bowl rendered silver
        if a.get("name"):
            a.set("name", f"{name}_{a.get('name')}")
        if a.get("texture"):
            a.set("texture", f"{name}_{a.get('texture')}")
    inner = o.find("worldbody").find("body").find("body")
    for g in inner.iter("geom"):
        if g.get("contype") != "0":  # object geoms touch the robot and the world
            g.set("contype", "1")
            g.set("conaffinity", "3")
        if g.get("group") == "0":  # collision boxes: group 0 is drawn by default and greyed the object over
            g.set("group", "3")
        if g.get("mesh"):
            g.set("mesh", f"{name}_{g.get('mesh')}")
        if g.get("material"):
            g.set("material", f"{name}_{g.get('material')}")
    inner.set("name", name)
    inner.set("pos", f"{pos[0]} {pos[1]} {pos[2]}")
    ET.SubElement(inner, "freejoint", {"name": f"{name}_free"})
    return assets, inner


def build(layout: Layout) -> str:
    """MJCF text for a layout."""
    rob = ET.parse(HERE / "xlerobot" / "xlerobot.xml").getroot()
    for mesh in rob.iter("mesh"):
        mesh.set("file", str(HERE / "xlerobot" / "meshes" / mesh.get("file")))
    # the base is parked at the table: no planar joints, no wheel joints
    for body in rob.iter("body"):
        for j in list(body.findall("joint")):
            if j.get("name") in ("root_x_axis_joint", "root_y_axis_joint", "root_z_rotation_joint", "wheel_right", "wheel_left"):
                body.remove(j)
    arm_joints = [f"{n}{s}" for s in SUFFIX.values() for n in ARM_JOINTS]
    jaws = [f"Jaw{s}" for s in SUFFIX.values()]
    for body in rob.iter("body"):
        nm = body.get("name", "")
        if nm in ("Right_Arm_Camera", "Left_Arm_Camera"):  # the URDF gives the link, not the look axis
            ET.SubElement(body, "camera", {"name": "right_wrist" if nm.startswith("Right") else "left_wrist",
                                           "pos": "0 0 0", "xyaxes": "0 -1 0 0 0 1", "fovy": "75"})
        if any(nm.startswith(k) for k in ("Rotation_Pitch", "Upper_Arm", "Lower_Arm", "Wrist_Pitch_Roll", "Fixed_Jaw", "Moving_Jaw")):
            body.set("gravcomp", "1")  # the servos hold their pose, as the STS3215s do
        for j in body.findall("joint"):
            if j.get("name") in arm_joints + jaws:
                j.set("damping", "2.0")
                j.set("armature", "0.01")
    for g in rob.iter("geom"):
        g.set("friction", "1.5 0.02 0.001")
        # robot parts never touch each other (a parked cart is welded to the world, which switches
        # MuJoCo's parent-child filter off), but they touch the table and the objects
        g.set("contype", "2")
        g.set("conaffinity", "1")

    scene = ET.Element("mujoco", {"model": "xlerobot_table"})
    ET.SubElement(scene, "compiler", {"angle": "radian"})  # the URDF's limits are radians
    ET.SubElement(scene, "option", {"timestep": "0.002", "integrator": "implicitfast", "cone": "elliptic", "impratio": "10"})
    vis = ET.SubElement(scene, "visual")
    ET.SubElement(vis, "global", {"offwidth": "640", "offheight": "480"})
    amb = f"{0.3 * layout.light:.3f}"
    ET.SubElement(vis, "headlight", {"ambient": f"{amb} {amb} {amb}", "diffuse": "0.35 0.35 0.35"})
    asset = ET.SubElement(scene, "asset")
    for a in rob.find("asset"):
        asset.append(a)
    ET.SubElement(asset, "texture", {"name": "grid", "type": "2d", "builtin": "checker", "rgb1": ".85 .85 .85",
                                     "rgb2": ".75 .75 .75", "width": "256", "height": "256"})
    ET.SubElement(asset, "material", {"name": "grid", "texture": "grid", "texrepeat": "8 8"})
    wb = ET.SubElement(scene, "worldbody")
    li = f"{layout.light:.3f}"
    ET.SubElement(wb, "light", {"pos": "0.4 0 2.5", "dir": "0 0 -1", "diffuse": f"{li} {li} {li}"})
    ET.SubElement(wb, "geom", {"name": "floor", "type": "plane", "size": "3 3 0.1", "material": "grid",
                               "contype": "1", "conaffinity": "3", "group": "1"})
    ET.SubElement(wb, "geom", {"name": "table", "type": "box", "pos": f"{FRONT + 0.30} 0 {TABLE_Z - 0.025}",
                               "size": "0.30 0.55 0.025", "rgba": " ".join(map(str, layout.table_rgb)) + " 1",
                               "friction": "1 0.005 0.0001", "contype": "1", "conaffinity": "3", "group": "1"})
    # the head camera: the URDF's head_camera_rgb_optical_frame position, aimed at the workspace
    head = np.array([-0.0879, 0.0208, 1.1674]) + np.array(layout.head_jitter)
    ET.SubElement(wb, "camera", {"name": "head", "pos": " ".join(f"{v:.4f}" for v in head),
                                 "xyaxes": _lookat_xyaxes(head, [FRONT + 0.08, 0.0, TABLE_Z]), "fovy": "58"})
    # for people, not for the policy: across the table, facing the robot
    ET.SubElement(wb, "camera", {"name": "front", "pos": f"{FRONT + 0.85} 0.25 {TABLE_Z + 0.45}",
                                 "xyaxes": _lookat_xyaxes([FRONT + 0.85, 0.25, TABLE_Z + 0.45], [FRONT + 0.05, 0.0, TABLE_Z + 0.08]),
                                 "fovy": "50"})
    for b in rob.find("worldbody"):
        wb.append(b)
    root = libero_objects()
    places = {"bowl": (*layout.bowl, TABLE_Z + 0.02), "plate": (*layout.plate, TABLE_Z + 0.01)}
    if layout.ramekin is not None:
        places["ramekin"] = (*layout.ramekin, TABLE_Z + 0.01)
    for name, pos in places.items():
        a_list, body = _object(name, root, pos)
        for a in a_list:
            asset.append(a)
        wb.append(body)
    act = ET.SubElement(scene, "actuator")
    for j in arm_joints:
        ET.SubElement(act, "position", {"name": f"a_{j}", "joint": j, "kp": "60", "kv": "4", "forcerange": "-8 8"})
    for j in jaws:
        ET.SubElement(act, "position", {"name": f"a_{j}", "joint": j, "kp": "8", "kv": "0.4", "forcerange": "-2.5 2.5"})
    return ET.tostring(scene, encoding="unicode")

