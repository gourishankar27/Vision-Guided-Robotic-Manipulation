"""Collect a bright Franka pick/place episode from NVIDIA Isaac Sim.

Run this with Isaac Sim's Python environment, not the lightweight training venv.
Example:
    python scripts/isaac_sim/franka_reacher_collect.py --headless --output-dir datasets/isaac_franka_bright

The v5 collector is intentionally conservative: it builds a visible table,
adds explicit USD lights, aims the camera with a look-at transform, checks frame
brightness, and stores only image/state pairs that are aligned.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect Franka RGB/trajectory data from Isaac Sim.")
    parser.add_argument("--output-dir", default="datasets/isaac_franka")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--num-steps", type=int, default=600)
    parser.add_argument("--save-every", type=int, default=4)
    parser.add_argument("--warmup-frames", type=int, default=24, help="Render frames before collection so RGB annotator is ready.")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--cube-x", type=float, default=0.35)
    parser.add_argument("--cube-y", type=float, default=0.25)
    parser.add_argument("--place-x", type=float, default=-0.25)
    parser.add_argument("--place-y", type=float, default=-0.30)
    parser.add_argument("--camera-view", choices=["diag", "overhead", "front", "side"], default="diag")
    parser.add_argument("--lighting", choices=["studio", "bright", "soft", "minimal"], default="studio")
    parser.add_argument("--renderer", default="RayTracedLighting", help="Renderer passed to SimulationApp when supported.")
    parser.add_argument("--min-brightness", type=float, default=25.0, help="Warn/rescue if preview mean RGB is below this 0-255 value.")
    parser.add_argument("--no-table", action="store_true", help="Disable the explicit light-gray table/platform.")
    parser.add_argument("--save-debug-preview", action="store_true", help="Save preview RGB before control starts.")
    parser.add_argument("--randomize-visuals", action="store_true", help="Randomize cube/table colors and light intensity using --seed.")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    return v / max(float(np.linalg.norm(v)), eps)


def _rotation_matrix_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """Convert a right-handed 3x3 rotation matrix to scalar-first quaternion."""
    R = np.asarray(R, dtype=np.float64)
    tr = float(np.trace(R))
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    else:
        i = int(np.argmax(np.diag(R)))
        if i == 0:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif i == 1:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    return _normalize(q).astype(np.float32)


def _look_at_quat_usd(eye: np.ndarray, target: np.ndarray, up_guess: np.ndarray = np.array([0.0, 0.0, 1.0])) -> np.ndarray:
    """Quaternion for a USD camera, whose local camera forward axis is -Z."""
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    forward = _normalize(target - eye)
    up_guess = _normalize(up_guess)
    # If nearly looking straight along up, use a different up vector.
    if abs(float(np.dot(forward, up_guess))) > 0.95:
        up_guess = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    back = -forward
    right = _normalize(np.cross(up_guess, back))
    up = _normalize(np.cross(back, right))
    # Columns are local X/Y/Z axes expressed in world coordinates.
    R = np.stack([right, up, back], axis=1)
    return _rotation_matrix_to_quat_wxyz(R)


def _camera_pose(view: str) -> Tuple[np.ndarray, np.ndarray]:
    target = np.array([0.08, 0.02, 0.18], dtype=np.float32)
    presets = {
        "diag": np.array([1.20, -1.20, 0.95], dtype=np.float32),
        "overhead": np.array([0.05, -0.03, 1.65], dtype=np.float32),
        "front": np.array([1.25, 0.02, 0.70], dtype=np.float32),
        "side": np.array([0.05, -1.35, 0.72], dtype=np.float32),
    }
    return presets[view], target


def _add_lighting(world, lighting: str, rng: np.random.Generator) -> dict:
    from pxr import Gf, UsdGeom, UsdLux

    try:
        stage = world.stage
    except Exception:
        import omni.usd

        stage = omni.usd.get_context().get_stage()
    UsdGeom.Xform.Define(stage, "/World/Lighting")

    factors = {
        "minimal": 0.6,
        "soft": 1.0,
        "studio": 1.8,
        "bright": 2.7,
    }
    f = factors.get(lighting, 1.8)
    # Mild randomization keeps the visual encoder from overfitting one exposure.
    f *= float(rng.uniform(0.85, 1.15))

    dome = UsdLux.DomeLight.Define(stage, "/World/Lighting/DomeLight")
    dome.CreateIntensityAttr(650.0 * f)
    dome.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))

    key = UsdLux.DistantLight.Define(stage, "/World/Lighting/KeyLight")
    key.CreateIntensityAttr(1800.0 * f)
    key.CreateAngleAttr(0.45)
    key.CreateColorAttr(Gf.Vec3f(1.0, 0.96, 0.90))
    UsdGeom.XformCommonAPI(key).SetRotate((55.0, 0.0, -35.0), UsdGeom.XformCommonAPI.RotationOrderXYZ)

    fill = UsdLux.RectLight.Define(stage, "/World/Lighting/FillRect")
    fill.CreateIntensityAttr(4500.0 * f)
    fill.CreateWidthAttr(3.0)
    fill.CreateHeightAttr(3.0)
    fill.CreateColorAttr(Gf.Vec3f(0.86, 0.92, 1.0))
    xform = UsdGeom.XformCommonAPI(fill)
    xform.SetTranslate((0.0, -1.2, 2.2))
    xform.SetRotate((65.0, 0.0, 0.0), UsdGeom.XformCommonAPI.RotationOrderXYZ)

    return {"lighting": lighting, "lighting_factor": f}


def _safe_set_camera_view(camera, eye: np.ndarray, target: np.ndarray) -> None:
    """Set camera pose using Isaac viewport helper when available, with a quaternion fallback."""
    quat = _look_at_quat_usd(eye, target)
    try:
        # Available in recent Isaac Sim. It handles view-axis details reliably.
        from isaacsim.core.utils.viewports import set_camera_view

        set_camera_view(eye=eye, target=target, camera_prim_path=camera.prim_path)
        return
    except Exception:
        pass
    # Fallback: set USD camera pose directly. Try the explicit USD axes mode first.
    try:
        camera.set_world_pose(position=eye, orientation=quat, camera_axes="usd")
    except Exception:
        camera.set_world_pose(position=eye, orientation=quat)


def _rgb_stats(rgb) -> dict:
    if rgb is None:
        return {"mean": -1.0, "std": -1.0, "min": -1.0, "max": -1.0}
    arr = np.asarray(rgb)
    if arr.ndim == 3:
        arr = arr[..., :3]
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(int(args.seed))
    out_dir = Path(args.output_dir)
    frame_dir = out_dir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)

    from isaacsim import SimulationApp

    app_cfg = {"headless": bool(args.headless), "width": args.width, "height": args.height}
    if args.renderer:
        app_cfg["renderer"] = args.renderer
    simulation_app = SimulationApp(app_cfg)

    import imageio.v2 as imageio
    import omni.kit.app
    from isaacsim.core.api import World
    from isaacsim.core.api.objects import DynamicCuboid
    from isaacsim.robot.manipulators.examples.franka import Franka
    from isaacsim.robot.manipulators.examples.franka.controllers import PickPlaceController
    from isaacsim.sensors.camera import Camera

    try:
        from isaacsim.core.api.objects import FixedCuboid
    except Exception:  # pragma: no cover - depends on Isaac Sim package version
        FixedCuboid = None

    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()
    lighting_meta = _add_lighting(world, args.lighting, rng)

    if not args.no_table and FixedCuboid is not None:
        table_color = np.array([0.78, 0.78, 0.74], dtype=np.float32)
        if args.randomize_visuals:
            table_color = rng.uniform([0.55, 0.55, 0.55], [0.90, 0.90, 0.86]).astype(np.float32)
        world.scene.add(
            FixedCuboid(
                prim_path="/World/work_table",
                name="work_table",
                position=np.array([0.05, 0.0, -0.025]),
                scale=np.array([1.25, 1.00, 0.05]),
                color=table_color,
            )
        )

    cube_size = 0.055
    cube_color = np.array([0.0, 0.8, 1.0], dtype=np.float32)
    if args.randomize_visuals:
        cube_color = rng.uniform([0.0, 0.35, 0.45], [0.25, 1.0, 1.0]).astype(np.float32)
    cube = world.scene.add(
        DynamicCuboid(
            prim_path="/World/target_cube",
            name="target_cube",
            position=np.array([args.cube_x, args.cube_y, cube_size / 2.0]),
            scale=np.array([cube_size, cube_size, cube_size]),
            color=cube_color,
        )
    )
    franka = world.scene.add(Franka(prim_path="/World/Franka", name="franka"))

    camera = Camera(
        prim_path="/World/Camera",
        position=np.array([1.2, -1.2, 0.95], dtype=np.float32),
        resolution=(args.width, args.height),
    )
    eye, cam_target = _camera_pose(args.camera_view)

    world.reset()
    camera.initialize()
    _safe_set_camera_view(camera, eye, cam_target)

    # The RGB annotator can return None for the first few rendered frames. Step Kit too.
    for _ in range(max(0, int(args.warmup_frames))):
        world.step(render=True)
        omni.kit.app.get_app().update()

    preview = camera.get_rgb()
    stats = _rgb_stats(preview)
    if args.save_debug_preview and preview is not None:
        imageio.imwrite(out_dir / "debug_preview.png", preview)
    if stats["mean"] < float(args.min_brightness):
        print(
            f"WARNING: dark preview mean={stats['mean']:.2f}. Trying overhead rescue camera and brighter lighting."
        )
        eye2, target2 = _camera_pose("overhead")
        _safe_set_camera_view(camera, eye2, target2)
        for _ in range(12):
            world.step(render=True)
            omni.kit.app.get_app().update()
        preview = camera.get_rgb()
        rescue_stats = _rgb_stats(preview)
        stats = {f"before_{k}": v for k, v in stats.items()} | {f"after_{k}": v for k, v in rescue_stats.items()}
        if args.save_debug_preview and preview is not None:
            imageio.imwrite(out_dir / "debug_preview_rescue.png", preview)
    print("Preview RGB stats:", json.dumps(stats, indent=2))

    controller = PickPlaceController(
        name="pick_place_controller",
        gripper=franka.gripper,
        robot_articulation=franka,
    )
    target_position = np.array([args.place_x, args.place_y, cube_size / 2.0])

    joint_positions = []
    cube_positions = []
    ee_positions = []
    frame_paths = []
    frame_steps = []
    frame_brightness = []

    for step in range(args.num_steps):
        cube_pos, _ = cube.get_world_pose()
        q = franka.get_joint_positions()
        action = controller.forward(
            picking_position=cube_pos,
            placing_position=target_position,
            current_joint_positions=q,
        )
        franka.apply_action(action)
        world.step(render=True)

        if step % args.save_every == 0:
            rgb = camera.get_rgb()
            if rgb is None:
                continue

            path = frame_dir / f"rgb_{step:06d}.png"
            imageio.imwrite(path, rgb)
            frame_paths.append(str(path.relative_to(out_dir)))
            frame_steps.append(step)
            frame_brightness.append(float(np.mean(np.asarray(rgb)[..., :3])))

            cube_pos, _ = cube.get_world_pose()
            try:
                ee_pos, _ = franka.end_effector.get_world_pose()
            except Exception:
                ee_pos = np.zeros(3, dtype=np.float32)
            joint_positions.append(franka.get_joint_positions().copy())
            cube_positions.append(np.asarray(cube_pos, dtype=np.float32).copy())
            ee_positions.append(np.asarray(ee_pos, dtype=np.float32).copy())

        if controller.is_done():
            break

    meta = {
        "camera_view": args.camera_view,
        "camera_eye": eye.tolist(),
        "camera_target": cam_target.tolist(),
        "lighting": lighting_meta,
        "width": args.width,
        "height": args.height,
        "mean_frame_brightness": float(np.mean(frame_brightness)) if frame_brightness else -1.0,
        "min_frame_brightness": float(np.min(frame_brightness)) if frame_brightness else -1.0,
        "max_frame_brightness": float(np.max(frame_brightness)) if frame_brightness else -1.0,
    }
    np.savez_compressed(
        out_dir / "episode.npz",
        joint_positions=np.asarray(joint_positions, dtype=np.float32),
        cube_positions=np.asarray(cube_positions, dtype=np.float32),
        end_effector_positions=np.asarray(ee_positions, dtype=np.float32),
        target_position=target_position.astype(np.float32),
        frame_paths=np.asarray(frame_paths),
        frame_steps=np.asarray(frame_steps, dtype=np.int32),
        frame_brightness=np.asarray(frame_brightness, dtype=np.float32),
        save_every=np.asarray(args.save_every),
        metadata=np.asarray(json.dumps(meta)),
    )
    with (out_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved Isaac Sim episode to {out_dir}")
    print(f"Saved aligned frames/states: {len(frame_paths)}")
    print(f"Mean saved brightness: {meta['mean_frame_brightness']:.2f}")
    simulation_app.close()


if __name__ == "__main__":
    main()
