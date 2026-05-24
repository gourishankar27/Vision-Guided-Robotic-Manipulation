"""Collect a simple Franka pick/place episode from NVIDIA Isaac Sim.

Run this with Isaac Sim's Python environment, not the lightweight training venv.
Example:
    python scripts/isaac_sim/franka_reacher_collect.py --headless --output-dir datasets/isaac_franka

The output is intentionally simple: RGB frames plus an episode.npz containing
joint positions, cube positions, end-effector positions, and frame paths. That
keeps the Isaac Sim side decoupled from the differentiable PyTorch/Warp code.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect Franka RGB/trajectory data from Isaac Sim.")
    parser.add_argument("--output-dir", default="datasets/isaac_franka")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--num-steps", type=int, default=600)
    parser.add_argument("--save-every", type=int, default=4)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--cube-x", type=float, default=0.35)
    parser.add_argument("--cube-y", type=float, default=0.25)
    parser.add_argument("--place-x", type=float, default=-0.25)
    parser.add_argument("--place-y", type=float, default=-0.30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    frame_dir = out_dir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)

    # Isaac Sim imports must happen after SimulationApp starts.
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": bool(args.headless), "width": args.width, "height": args.height})

    import imageio.v2 as imageio
    import isaacsim.core.utils.numpy.rotations as rot_utils
    from isaacsim.core.api import World
    from isaacsim.core.api.objects import DynamicCuboid
    from isaacsim.robot.manipulators.examples.franka import Franka
    from isaacsim.robot.manipulators.examples.franka.controllers import PickPlaceController
    from isaacsim.sensors.camera import Camera

    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()

    cube_size = 0.055
    cube = world.scene.add(
        DynamicCuboid(
            prim_path="/World/target_cube",
            name="target_cube",
            position=np.array([args.cube_x, args.cube_y, cube_size / 2.0]),
            scale=np.array([cube_size, cube_size, cube_size]),
            color=np.array([0.0, 0.8, 1.0]),
        )
    )
    franka = world.scene.add(Franka(prim_path="/World/Franka", name="franka"))
    camera = Camera(
        prim_path="/World/Camera",
        position=np.array([1.25, -1.25, 0.95]),
        orientation=rot_utils.euler_angles_to_quats(np.array([62.0, 0.0, 43.0]), degrees=True),
        resolution=(args.width, args.height),
    )

    world.reset()
    camera.initialize()
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
            if rgb is not None:
                path = frame_dir / f"rgb_{step:06d}.png"
                imageio.imwrite(path, rgb)
                frame_paths.append(str(path.relative_to(out_dir)))
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

    np.savez_compressed(
        out_dir / "episode.npz",
        joint_positions=np.asarray(joint_positions, dtype=np.float32),
        cube_positions=np.asarray(cube_positions, dtype=np.float32),
        end_effector_positions=np.asarray(ee_positions, dtype=np.float32),
        target_position=target_position.astype(np.float32),
        frame_paths=np.asarray(frame_paths),
        save_every=np.asarray(args.save_every),
    )
    print(f"Saved Isaac Sim episode to {out_dir}")
    simulation_app.close()


if __name__ == "__main__":
    main()
