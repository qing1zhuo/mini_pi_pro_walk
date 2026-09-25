# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2021 ETH Zurich, Nikita Rudin
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2024 Beijing RobotEra TECHNOLOGY CO.,LTD. All rights reserved.

import csv
import os
from datetime import datetime
from statistics import mean

from isaacgym import gymapi
from isaacgym.torch_utils import *

import cv2
import numpy as np
import torch
from tqdm import tqdm

from humanoid import LEGGED_GYM_ROOT_DIR
from humanoid.envs import *
from humanoid.utils import (export_policy_as_jit, export_policy_to_onnx,
                            get_args, get_load_path, set_seed, task_registry)
from humanoid.utils.helpers import update_cfg_from_args
from humanoid.utils.math import wrap_to_pi

# x/y are body-frame velocity commands; heading is a world-frame target angle.
COMMANDS = (
    ("stand", 0.0, 0.0, 0.0),
    ("forward_slow", 0.3, 0.0, 0.0),
    ("forward_fast", 0.6, 0.0, 0.0),
    ("backward", -0.3, 0.0, 0.0),
    ("left", 0.0, 0.3, 0.0),
    ("right", 0.0, -0.3, 0.0),
    ("turn_left", 0.0, 0.0, np.pi / 2),
    ("turn_right", 0.0, 0.0, -np.pi / 2),
    ("turn_around", 0.0, 0.0, np.pi),
    ("forward_turn_left", 0.3, 0.0, np.pi / 2),
    ("forward_turn_right", 0.3, 0.0, -np.pi / 2),
)


def _reset_eval_state(env, seed):
    set_seed(seed)
    env.reset_idx(torch.arange(env.num_envs, device=env.device))
    env.last_root_vel.zero_()
    env.last_contacts.zero_()
    env.feet_height.zero_()
    env.last_feet_z = 0.05
    env.rand_push_force.zero_()
    env.rand_push_torque.zero_()
    env.base_lin_vel[:] = quat_rotate_inverse(env.base_quat, env.root_states[:, 7:10])
    env.base_ang_vel[:] = quat_rotate_inverse(env.base_quat, env.root_states[:, 10:13])


def _set_command(env, vx, vy, heading):
    env.commands[:, 0] = vx
    env.commands[:, 1] = vy
    env.commands[:, 3] = heading
    env.commands[:, 2] = torch.clamp(
        0.5 * wrap_to_pi(heading - env.base_euler_xyz[:, 2]), -1.0, 1.0
    )


def _scheduled_push(env, seed, case_name, step, active, writer):
    with torch.random.fork_rng(
        devices=[env.sim_device_id] if env.device != "cpu" else []
    ):
        torch.manual_seed(seed + 100000 + step)
        env._push_robots()
    samples = torch.cat(
        (env.rand_push_force[:, :2], env.rand_push_torque), dim=1
    ).cpu().tolist()
    active_ids = active.cpu().tolist()
    writer.writerows(
        [case_name, step, i, int(active_ids[i]), *sample]
        for i, sample in enumerate(samples)
    )


def _evaluate_fixed_case(
    env,
    policy,
    case,
    seed,
    scheduled_push,
    warmup_steps,
    push_interval,
    recovery_threshold,
    friction,
    base_mass,
    episodes,
    pushes,
    reward_names,
    render,
    video_dir,
    camera,
):
    name, vx, vy, heading = case
    _reset_eval_state(env, seed + 1)
    _set_command(env, vx, vy, heading)
    env.compute_observations()
    obs = env.get_observations()

    active = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)
    steps = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    timeouts = torch.zeros_like(active)
    reward = torch.zeros(env.num_envs, device=env.device)
    errors = torch.zeros(env.num_envs, 4, device=env.device)
    max_roll_pitch = torch.zeros(env.num_envs, 2, device=env.device)
    had_push = torch.zeros_like(active)
    pending_recovery = torch.zeros_like(active)
    recovery_start = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    recovery_total = torch.zeros(env.num_envs, device=env.device)
    recovery_count = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    recovery_missed = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    action_delta_sum = torch.zeros(env.num_envs, device=env.device)
    action_delta_max = torch.zeros(env.num_envs, device=env.device)
    torque_peak = torch.zeros(env.num_envs, device=env.device)
    saturation_count = torch.zeros(env.num_envs, device=env.device)
    last_actions = None
    torque_history = []
    terms = {}

    video = None
    if render:
        video_path = os.path.join(video_dir, name + ".mp4")
        video = cv2.VideoWriter(
            video_path, cv2.VideoWriter_fourcc(*"mp4v"), 1.0 / env.dt, (720, 480)
        )
        if not video.isOpened():
            raise RuntimeError("Cannot open video output: " + video_path)
    try:
        for step in tqdm(range(int(env.max_episode_length) + 2), desc=name):
            if (
                scheduled_push
                and step >= warmup_steps
                and (step - warmup_steps) % push_interval == 0
            ):
                recovery_missed += (pending_recovery & active).long()
                _scheduled_push(env, seed, name, step, active, pushes)
                had_push |= active
                pending_recovery |= active
                recovery_start[active] = step

            command = env.commands[:, :3].clone()
            heading_error = torch.abs(
                wrap_to_pi(heading - env.base_euler_xyz[:, 2])
            )
            with torch.no_grad():
                actions = policy(obs.detach())
            was_active = active.clone()
            if last_actions is not None:
                delta = torch.abs(actions - last_actions)
                action_delta_sum[was_active] += delta[was_active].sum(dim=1)
                action_delta_max[was_active] = torch.maximum(
                    action_delta_max[was_active], delta[was_active].amax(dim=1)
                )
            last_actions = actions.clone()
            obs, _, rews, dones, infos = env.step(actions.detach())

            steps[was_active] += 1
            reward[was_active] += rews[was_active]
            errors[was_active, 0] += torch.abs(
                env.base_lin_vel[was_active, 0] - command[was_active, 0]
            )
            errors[was_active, 1] += torch.abs(
                env.base_lin_vel[was_active, 1] - command[was_active, 1]
            )
            errors[was_active, 2] += torch.abs(
                env.base_ang_vel[was_active, 2] - command[was_active, 2]
            )
            errors[was_active, 3] += heading_error[was_active]

            torque_abs = torch.abs(env.torques)
            torque_history.append(torque_abs.clone())
            torque_peak[was_active] = torch.maximum(
                torque_peak[was_active], torque_abs[was_active].amax(dim=1)
            )
            saturation_count[was_active] += (
                torque_abs[was_active] >= 0.999 * env.torque_limits
            ).sum(dim=1)

            still_running = was_active & ~dones.bool()
            pushed_running = still_running & had_push
            max_roll_pitch[pushed_running] = torch.maximum(
                max_roll_pitch[pushed_running],
                torch.abs(env.base_euler_xyz[pushed_running, :2]),
            )
            speed_error = torch.amax(
                torch.abs(
                    torch.stack(
                        (
                            env.base_lin_vel[:, 0] - command[:, 0],
                            env.base_lin_vel[:, 1] - command[:, 1],
                            env.base_ang_vel[:, 2] - command[:, 2],
                        ),
                        dim=1,
                    )
                ),
                dim=1,
            )
            recovered = pending_recovery & still_running & (speed_error <= recovery_threshold)
            recovery_total[recovered] += (step - recovery_start[recovered] + 1) * env.dt
            recovery_count[recovered] += 1
            pending_recovery[recovered] = False

            finished = was_active & dones.bool()
            if torch.any(finished):
                recovery_missed += (pending_recovery & finished).long()
                pending_recovery[finished] = False
                timeouts[finished] = infos["time_outs"][finished]
                finished_ids = set(finished.nonzero(as_tuple=True)[0].tolist())
                for position, env_id in enumerate(env.eval_episode_ids.tolist()):
                    if env_id in finished_ids:
                        terms[env_id] = {
                            key: float(values[position])
                            for key, values in env.eval_episode_rewards.items()
                        }
            if video is not None and was_active[0] and not dones[0]:
                env.gym.fetch_results(env.sim, True)
                env.gym.step_graphics(env.sim)
                env.gym.render_all_camera_sensors(env.sim)
                image = env.gym.get_camera_image(
                    env.sim, env.envs[0], camera, gymapi.IMAGE_COLOR
                )
                video.write(
                    cv2.cvtColor(
                        np.reshape(image, (480, 720, 4)), cv2.COLOR_RGBA2BGR
                    )
                )
            active &= ~dones.bool()
            if not torch.any(active):
                break
    finally:
        if video is not None:
            video.release()

    if torch.any(active):
        raise RuntimeError(
            f"{name}: some environments did not finish within "
            f"{int(env.max_episode_length) + 2} steps"
        )

    torques = torch.stack(torque_history)
    counts = steps.cpu().tolist()
    totals = reward.cpu().tolist()
    error_values = errors.cpu().tolist()
    timeout_values = timeouts.cpu().tolist()
    for i in range(env.num_envs):
        count = counts[i]
        torque_p95 = float(torch.quantile(torques[:count, i].flatten(), 0.95))
        recovered_count = int(recovery_count[i])
        row = {
            "case": name,
            "env_id": i,
            "command_vx": vx,
            "command_vy": vy,
            "target_heading": heading,
            "friction": friction[i],
            "base_mass": base_mass[i],
            "steps": count,
            "seconds": count * env.dt,
            "timeout": int(timeout_values[i]),
            "total_reward": totals[i],
            "vx_mae": error_values[i][0] / count,
            "vy_mae": error_values[i][1] / count,
            "yaw_rate_mae": error_values[i][2] / count,
            "heading_mae": error_values[i][3] / count,
            "max_abs_roll_after_push": float(max_roll_pitch[i, 0]),
            "max_abs_pitch_after_push": float(max_roll_pitch[i, 1]),
            "mean_recovery_time_s": (
                float(recovery_total[i]) / recovered_count if recovered_count else ""
            ),
            "recovered_pushes": recovered_count,
            "unrecovered_pushes": int(recovery_missed[i]),
            "early_termination_after_push": int(
                bool(had_push[i].item()) and not bool(timeouts[i].item())
            ),
            "peak_torque": float(torque_peak[i]),
            "p95_torque": torque_p95,
            "torque_saturation_ratio": float(saturation_count[i])
            / (count * env.num_actions),
            "mean_abs_action_delta": float(action_delta_sum[i])
            / (max(count - 1, 1) * env.num_actions),
            "max_abs_action_delta": float(action_delta_max[i]),
        }
        row.update({"reward_" + key: value for key, value in terms.get(i, {}).items()})
        episodes.writerow(row)
        reward_names.update(terms.get(i, {}).keys())
        yield row


def _evaluate_switches(
    env,
    policy,
    commands,
    periods,
    seed,
    recovery_threshold,
    writer,
):
    for period_s in periods:
        period_steps = max(1, int(round(period_s / env.dt)))
        _reset_eval_state(env, seed + 1)
        command_index = 0
        _set_command(env, *commands[command_index][1:])
        env.compute_observations()
        obs = env.get_observations()
        active = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)
        pending = [None] * env.num_envs
        events = []

        for step in tqdm(
            range(int(env.max_episode_length) + 2), desc=f"switch_{period_s:g}s"
        ):
            if step > 0 and step % period_steps == 0:
                previous = commands[command_index]
                command_index = (command_index + 1) % len(commands)
                current = commands[command_index]
                _set_command(env, *current[1:])
                env.compute_observations()
                obs = env.get_observations()
                for env_id in active.nonzero(as_tuple=True)[0].tolist():
                    if pending[env_id] is not None:
                        pending[env_id]["recovery_time_s"] = ""
                    event = {
                        "period_s": period_s,
                        "env_id": env_id,
                        "switch_step": step,
                        "switch_time_s": step * env.dt,
                        "before_case": previous[0],
                        "before_vx": previous[1],
                        "before_vy": previous[2],
                        "before_heading": previous[3],
                        "after_case": current[0],
                        "after_vx": current[1],
                        "after_vy": current[2],
                        "after_heading": current[3],
                        "recovery_time_s": "",
                        "terminated_before_recovery": 0,
                    }
                    events.append(event)
                    pending[env_id] = event

            command = env.commands[:, :3].clone()
            with torch.no_grad():
                actions = policy(obs.detach())
            was_active = active.clone()
            obs, _, _, dones, _ = env.step(actions.detach())
            still_running = was_active & ~dones.bool()
            speed_error = torch.amax(
                torch.abs(
                    torch.stack(
                        (
                            env.base_lin_vel[:, 0] - command[:, 0],
                            env.base_lin_vel[:, 1] - command[:, 1],
                            env.base_ang_vel[:, 2] - command[:, 2],
                        ),
                        dim=1,
                    )
                ),
                dim=1,
            )
            for env_id in (
                still_running & (speed_error <= recovery_threshold)
            ).nonzero(as_tuple=True)[0].tolist():
                event = pending[env_id]
                if event is not None:
                    event["recovery_time_s"] = (
                        step - event["switch_step"] + 1
                    ) * env.dt
                    pending[env_id] = None
            for env_id in (was_active & dones.bool()).nonzero(as_tuple=True)[0].tolist():
                if pending[env_id] is not None:
                    pending[env_id]["terminated_before_recovery"] = 1
                    pending[env_id] = None
            active &= ~dones.bool()
            if not torch.any(active):
                break

        for event in events:
            writer.writerow(event)


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    evaluation_cfg = getattr(env_cfg, "evaluation", None)
    if evaluation_cfg is not None:
        commands = tuple(evaluation_cfg.cases)
        interpolation_cases = tuple(evaluation_cfg.interpolation_cases)
        switch_periods = tuple(evaluation_cfg.command_switch_periods_s)
        scheduled_push = evaluation_cfg.scheduled_push
        render = evaluation_cfg.render
        export_policy = evaluation_cfg.export_policy
        default_seed = evaluation_cfg.seed
        push_warmup_s = evaluation_cfg.push_warmup_s
    else:
        commands = COMMANDS
        interpolation_cases = ()
        switch_periods = ()
        scheduled_push = True
        render = globals().get("RENDER", True)
        export_policy = globals().get("EXPORT_POLICY", True)
        default_seed = 123145
        push_warmup_s = 0.0

    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 64)
    env_cfg.sim.max_gpu_contact_pairs = 2**10
    env_cfg.terrain.mesh_type = "plane"
    if evaluation_cfg is None:
        env_cfg.noise.add_noise = True
        env_cfg.noise.curriculum = False
        env_cfg.noise.noise_level = 0.5
        env_cfg.domain_rand.randomize_friction = True
    env_cfg.commands.resampling_time = env_cfg.env.episode_length_s + 1.0
    env_cfg.domain_rand.push_robots = False
    seed = args.seed if args.seed is not None else default_seed
    env_cfg.seed = train_cfg.seed = seed
    _, train_cfg = update_cfg_from_args(None, train_cfg, args)
    train_cfg.runner.resume = True
    log_root = os.path.join(
        LEGGED_GYM_ROOT_DIR, "logs", train_cfg.runner.experiment_name
    )
    checkpoint = get_load_path(
        log_root, train_cfg.runner.load_run, train_cfg.runner.checkpoint
    )

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    if not args.headless:
        env.set_camera(env_cfg.viewer.pos, env_cfg.viewer.lookat)
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg
    )
    policy = ppo_runner.get_inference_policy(device=env.device)

    if export_policy:
        path = os.path.join(log_root, "exported", "policies")
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        export_policy_to_onnx(ppo_runner.alg.actor_critic, path)
        print("Exported policy to:", path)

    output_dir = os.path.join(
        os.path.dirname(checkpoint),
        "eval",
        datetime.now().strftime("%Y%m%d_%H%M%S_%f") + f"_seed{seed}",
    )
    os.makedirs(output_dir, exist_ok=True)
    video_dir = os.path.join(output_dir, "videos")
    if render:
        os.makedirs(video_dir, exist_ok=True)
    env.record_eval_rewards = True
    friction = env.friction_coeffs.view(-1).tolist()
    base_mass = [
        env.gym.get_actor_rigid_body_properties(
            env.envs[i], env.gym.get_actor_handle(env.envs[i], 0)
        )[0].mass
        for i in range(env.num_envs)
    ]
    reward_names = set(env.episode_sums)
    fields = [
        "case", "env_id", "command_vx", "command_vy", "target_heading",
        "friction", "base_mass", "steps", "seconds", "timeout",
        "total_reward", "vx_mae", "vy_mae", "yaw_rate_mae", "heading_mae",
        "max_abs_roll_after_push", "max_abs_pitch_after_push",
        "mean_recovery_time_s", "recovered_pushes", "unrecovered_pushes",
        "early_termination_after_push", "peak_torque", "p95_torque",
        "torque_saturation_ratio", "mean_abs_action_delta",
        "max_abs_action_delta",
    ]
    fields += ["reward_" + name for name in sorted(reward_names)]
    switch_fields = [
        "period_s", "env_id", "switch_step", "switch_time_s", "before_case",
        "before_vx", "before_vy", "before_heading", "after_case", "after_vx",
        "after_vy", "after_heading", "recovery_time_s",
        "terminated_before_recovery",
    ]
    push_interval = max(
        1, int(round(env.cfg.domain_rand.push_interval_s / env.dt))
    )
    warmup_steps = int(round(push_warmup_s / env.dt))
    recovery_threshold = 0.1
    all_rows = []

    camera = None
    if render:
        camera = env.camera_handle
        offset = gymapi.Vec3(1, -1, 0.5)
        rotation = gymapi.Quat.from_axis_angle(
            gymapi.Vec3(-0.3, 0.2, 1), np.deg2rad(135)
        )
        actor = env.gym.get_actor_handle(env.envs[0], 0)
        body = env.gym.get_actor_rigid_body_handle(env.envs[0], actor, 0)
        env.gym.attach_camera_to_body(
            camera,
            env.envs[0],
            body,
            gymapi.Transform(offset, rotation),
            gymapi.FOLLOW_POSITION,
        )

    with open(
        os.path.join(output_dir, "episodes.csv"), "w", newline="", encoding="utf-8"
    ) as episodes_file, open(
        os.path.join(output_dir, "pushes.csv"), "w", newline="", encoding="utf-8"
    ) as pushes_file, open(
        os.path.join(output_dir, "command_switches.csv"),
        "w",
        newline="",
        encoding="utf-8",
    ) as switches_file:
        episodes = csv.DictWriter(episodes_file, fieldnames=fields)
        episodes.writeheader()
        pushes = csv.writer(pushes_file)
        pushes.writerow(
            [
                "case", "step", "env_id", "active", "push_vx", "push_vy",
                "push_wx", "push_wy", "push_wz",
            ]
        )
        switches = csv.DictWriter(switches_file, fieldnames=switch_fields)
        switches.writeheader()

        for case in commands + interpolation_cases:
            rows = _evaluate_fixed_case(
                env, policy, case, seed, scheduled_push, warmup_steps,
                push_interval, recovery_threshold, friction, base_mass, episodes,
                pushes, reward_names, render, video_dir, camera,
            )
            all_rows.extend(rows)
            episodes_file.flush()
            pushes_file.flush()

        if switch_periods:
            _evaluate_switches(
                env, policy, commands, switch_periods, seed,
                recovery_threshold, switches,
            )

    with open(
        os.path.join(output_dir, "report.txt"), "w", encoding="utf-8"
    ) as report:
        report.write(
            f"Checkpoint: {checkpoint}\nSeed: {seed}\nTask: {args.task}\n"
            f"Environments: {env.num_envs}\n"
        )
        report.write(
            f"Episode limit: {env.cfg.env.episode_length_s}s; policy dt: {env.dt}s\n"
        )
        report.write(
            f"Friction range: {env.cfg.domain_rand.friction_range}; "
            "fixed when each environment is created.\n"
        )
        report.write(
            f"Scheduled pushes: {scheduled_push}; warmup={push_warmup_s}s; "
            f"interval={env.cfg.domain_rand.push_interval_s}s; "
            f"xy={env.cfg.domain_rand.max_push_vel_xy}; "
            f"angular={env.cfg.domain_rand.max_push_ang_vel}.\n"
        )
        report.write(
            f"Velocity recovery threshold: max component error <= "
            f"{recovery_threshold}.\n"
        )
        report.write(
            "Timeout=1 means the episode reached its time limit; timeout=0 means "
            "early termination. Push details are in pushes.csv and command switch "
            "recovery is in command_switches.csv.\n"
        )
        report.write(f"Videos: {video_dir if render else 'disabled'}\n\n")
        for name, vx, vy, heading in commands + interpolation_cases:
            rows = [row for row in all_rows if row["case"] == name]
            report.write(
                f"{name}: vx={vx}, vy={vy}, target_heading={heading:.3f}; "
                f"timeout_rate={mean(row['timeout'] for row in rows):.3f}, "
                f"mean_reward={mean(row['total_reward'] for row in rows):.3f}, "
                f"vx_mae={mean(row['vx_mae'] for row in rows):.3f}, "
                f"vy_mae={mean(row['vy_mae'] for row in rows):.3f}, "
                f"yaw_rate_mae={mean(row['yaw_rate_mae'] for row in rows):.3f}, "
                f"heading_mae={mean(row['heading_mae'] for row in rows):.3f}\n"
            )
    print("Evaluation saved to:", output_dir)


if __name__ == "__main__":
    EXPORT_POLICY = True
    RENDER = True
    args = get_args()
    play(args)
