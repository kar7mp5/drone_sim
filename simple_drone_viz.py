#!/usr/bin/env python3
import argparse
import time

import numpy as np
import pybullet as p

from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.utils.utils import sync


def run(gui: bool = True, duration_sec: float = 8.0) -> None:
    init_xyzs = np.array([[0.0, 0.0, 0.1]])
    init_rpys = np.array([[0.0, 0.0, 0.0]])

    env = CtrlAviary(
        drone_model=DroneModel("cf2x"),
        num_drones=1,
        initial_xyzs=init_xyzs,
        initial_rpys=init_rpys,
        physics=Physics("pyb"),
        pyb_freq=240,
        ctrl_freq=48,
        gui=gui,
        record=False,
        obstacles=True,
        user_debug_gui=False,
    )

    ctrl = DSLPIDControl(drone_model=DroneModel("cf2x"))
    pyb_client = env.getPyBulletClient()
    if gui:
        p.resetDebugVisualizerCamera(
            cameraDistance=1.8,
            cameraYaw=35,
            cameraPitch=-30,
            cameraTargetPosition=[0, 0, 0.6],
            physicsClientId=pyb_client,
        )

    action = np.zeros((1, 4))
    trail = []
    start = time.time()
    total_steps = int(duration_sec * env.CTRL_FREQ)

    for i in range(total_steps):
        obs, _, _, _, _ = env.step(action)

        # 0~2s: takeoff, 2s~: hover with a tiny circle
        t = i / env.CTRL_FREQ
        target_z = min(1.0, 0.1 + 0.45 * t)
        target_xy = np.array([0.0, 0.0])
        if t > 2.0:
            r = 0.25
            w = 0.7
            target_xy = np.array([r * np.cos(w * (t - 2.0)), r * np.sin(w * (t - 2.0))])

        target_pos = np.array([target_xy[0], target_xy[1], target_z])
        action[0, :], _, _ = ctrl.computeControlFromState(
            control_timestep=env.CTRL_TIMESTEP,
            state=obs[0],
            target_pos=target_pos,
            target_rpy=init_rpys[0],
        )

        if gui:
            pos = obs[0][0:3]
            trail.append(pos.copy())
            if len(trail) > 1:
                p.addUserDebugLine(
                    trail[-2],
                    trail[-1],
                    lineColorRGB=[0.2, 0.8, 1.0],
                    lineWidth=2.0,
                    lifeTime=0,
                    physicsClientId=pyb_client,
                )
            if i % 24 == 0:
                p.addUserDebugText(
                    f"t={t:.1f}s z={pos[2]:.2f}",
                    [0.2, -0.2, 1.3],
                    textColorRGB=[1, 1, 0.2],
                    textSize=1.2,
                    lifeTime=0.2,
                    physicsClientId=pyb_client,
                )
            sync(i, start, env.CTRL_TIMESTEP)

        env.render()

    final_pos = obs[0][0:3]
    print(f"Final drone position: {final_pos}")
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple drone visualization in PyBullet")
    parser.add_argument("--gui", action="store_true", help="Run with PyBullet GUI")
    parser.add_argument("--seconds", type=float, default=8.0, help="Duration in seconds")
    args = parser.parse_args()
    run(gui=args.gui, duration_sec=args.seconds)
