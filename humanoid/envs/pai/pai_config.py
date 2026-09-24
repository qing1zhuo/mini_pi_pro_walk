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


from humanoid.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

# 环境配置
class PaiCfg(LeggedRobotCfg):
    """Pi 机器人任务配置，覆盖 LeggedRobotCfg 的通用默认值。

    这里只声明参数；实际行为在 pai_env.py、legged_robot.py 和 terrain.py 中。
    BaseConfig 会把嵌套类实例化，因此运行时通过 cfg.env / cfg.rewards 等访问。
    改观测、动作或机器人模型时，还必须同步修改环境实现与部署端。
    """

    class env(LeggedRobotCfg.env):
        # Actor 每帧 47 维，保留最近 15 帧并展平为 705 维；无 RNN。
        # Critic 每帧 73 维，保留最近 3 帧并展平为 219 维特权观测。
        # 维度必须与 PaiFreeEnv.compute_observations() 的拼接结果一致。
        frame_stack = 15
        c_frame_stack = 3
        num_single_obs = 47
        num_observations = int(frame_stack * (num_single_obs))
        single_num_privileged_obs = 73
        num_privileged_obs = int(c_frame_stack * (single_num_privileged_obs))
        # 每条腿 6 个关节；--num_envs 可在训练命令中覆盖并行环境数。
        num_actions = 12
        num_envs = 4096
        # 最长回合 12 秒，按当前 0.02 秒策略步长约为 600 步。
        episode_length_s = 12
        # True 时把参考动作叠到策略动作上；当前只用参考轨迹塑造奖励。
        use_ref_actions = False

    class safety:
        # 位置/速度系数仅用于记录 URDF 上限；当前 PD 输出实际按力矩上限裁剪。
        pos_limit = 1.0
        vel_limit = 1.0
        # 最大关节力矩 = URDF effort × 0.85。
        torque_limit = 0.85

    class asset(LeggedRobotCfg.asset):
        # Isaac Gym 加载的 URDF；换机器人时还须检查关节顺序、观测和奖励索引。
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/pi_12dof_release_v1/urdf/pi_12dof_release_v1_rl.urdf"

        name = "Pai"
        # 用刚体名称子串寻找两只脚与两处膝盖。
        foot_name = "ankle_roll"
        knee_name = "calf"

        # 底座发生接触时计碰撞惩罚，并触发非超时终止。
        terminate_after_contacts_on = ["base_link"]
        penalize_contacts_on = ["base_link"]
        # Isaac Gym 的位标志：0 允许自碰撞，1 禁用。
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
        replace_cylinder_with_capsule = False
        # False 表示机器人根节点自由运动。
        fix_base_link = False

    class terrain(LeggedRobotCfg.terrain):
        # 当前是平地；下方网格尺寸、地形比例与课程参数暂不参与生成。
        mesh_type = "plane"
        # mesh_type = 'trimesh'
        curriculum = False
        # 不提供周围地面高度观测。若改为 True，须同步检查观测拼接和维度。
        measure_heights = False
        # 平地材质；机器人各碰撞形状的摩擦还会被 domain_rand 随机化。
        static_friction = 0.6
        dynamic_friction = 0.6
        # 以下只在 heightfield/trimesh 模式下用于地形块生成。
        terrain_length = 8.0
        terrain_width = 8.0
        num_rows = 20  # number of terrain rows (levels)
        num_cols = 20  # number of terrain cols (types)
        max_init_terrain_level = 10  # starting curriculum state
        # HumanoidTerrain.make_terrain 按比例累计值选择类型；平地模式不读取此列表。
        terrain_proportions = [0.2, 0.2, 0.4, 0.1, 0.1, 0, 0]
        # 接触恢复系数，0 表示不主动反弹。
        restitution = 0.0

    class noise:
        # 只对 Actor 的每帧观测加噪，再把带噪观测放入 15 帧历史；Critic 不加噪。
        add_noise = True  # False 时直接使用原观测，不读取下方噪声强度
        # 实际噪声 = 标准正态样本 × noise_level × 分量系数 × 对应观测缩放。
        noise_level = 0.6  # 全局噪声倍率；实现使用 torch.randn_like() 高斯采样

        class noise_scales:
            dof_pos = 0.05  # 关节角 12 维；再乘 obs_scales.dof_pos
            dof_vel = 0.5  # 关节速度 12 维；再乘 obs_scales.dof_vel
            ang_vel = 0.1  # 机身角速度 3 维；再乘 obs_scales.ang_vel
            lin_vel = 0.05  # 当前 Actor 无实际机身线速度，_get_noise_scale_vec() 不读取此项
            quat = 0.03  # 名称沿用 quat，实际给机身欧拉角 3 维加噪
            height_measurements = 0.1  # 当前不测地形高度，噪声向量也没有高度分量

    class init_state(LeggedRobotCfg.init_state):
        # 每次重置时的机身基准高度；其余姿态/速度沿用基类默认值。
        pos = [0.0, 0.0, 0.3453]
        # rot = [0., 0.27154693695611287, 0., 0.962425197628238]
        # 动作 0 时的 12 个关节目标角；key 必须与 URDF 的关节名称一致。
        default_joint_angles = {  # 单位 rad
            "r_hip_pitch_joint": 0.0,
            "r_hip_roll_joint": 0.0,
            "r_thigh_joint": 0.0,
            "r_calf_joint": 0.0,
            "r_ankle_pitch_joint": 0.0,
            "r_ankle_roll_joint": 0.0,
            "l_hip_pitch_joint": 0.0,
            "l_hip_roll_joint": 0.0,
            "l_thigh_joint": 0.0,
            "l_calf_joint": 0.0,
            "l_ankle_pitch_joint": 0.0,
            "l_ankle_roll_joint": 0.0,
        }

    class control(LeggedRobotCfg.control):
        # 键是关节名称子串，同一组参数同时匹配左右腿；单位分别为 N·m/rad 和 N·m·s/rad。
        stiffness = {
            "hip_pitch_joint": 40.0,
            "hip_roll_joint": 20.0,
            "thigh_joint": 20.0,
            "calf_joint": 40.0,
            "ankle_pitch_joint": 40.0,
            "ankle_roll_joint": 20.0,
        }
        damping = {
            "hip_pitch_joint": 0.6,
            "hip_roll_joint": 0.4,
            "thigh_joint": 0.4,
            "calf_joint": 0.6,
            "ankle_pitch_joint": 0.6,
            "ankle_roll_joint": 0.4,
        }
        # 目标关节角 = 默认角 + 0.25 × 策略动作，再由 PD 算出关节力矩。
        action_scale = 0.25
        # action_scale = 0.5
        # 每个策略动作维持 20 个物理步；0.001 × 20 = 0.02 秒，即 50 Hz。
        # decimation = 10  # 100hz
        decimation = 20  # 50 Hz 策略频率；物理仿真仍为 1000 Hz

    class sim(LeggedRobotCfg.sim):
        # PhysX 物理步长、子步数和竖直轴；up_axis=1 在本项目表示 z 轴向上。
        dt = 0.001  # 1000 Hz
        substeps = 1  # 2
        up_axis = 1  # 0 is y, 1 is z

        class physx(LeggedRobotCfg.sim.physx):
            # CPU 求解线程数与 TGS 求解器；线程数不等于并行环境数。
            num_threads = 30
            solver_type = 1  # 0: pgs, 1: tgs
            # 每个物理步的约束迭代次数。
            num_position_iterations = 4
            num_velocity_iterations = 0
            # 接触生成/分离的距离阈值，以及碰撞后的速度限制。
            contact_offset = 0.01  # [m]
            rest_offset = 0.0  # [m]
            bounce_threshold_velocity = 0.1  # [m/s]
            max_depenetration_velocity = 1.0
            # GPU 接触对容量和缓冲区倍率，环境数增加时可能需要调大。
            max_gpu_contact_pairs = 2**23  # 2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            # 接触力采集：0 不采集，1 只采末子步，2 采全部子步；奖励和终止依赖接触力。
            contact_collection = 2

    class domain_rand:
        # 创建环境时按环境随机设置机器人碰撞形状的摩擦系数。
        randomize_friction = True
        friction_range = [0.1, 2.0]
        # 创建环境时给底座刚体质量加一个均匀采样的 kg 偏移。
        randomize_base_mass = True
        added_mass_range = [-1.0, 1.0]
        # 每 4 秒直接改写机身速度，模拟外部推动；不是施加真实外力。
        push_robots = True
        push_interval_s = 4
        # 线速度扰动作用在 x/y，角速度扰动作用在 x/y/z。
        max_push_vel_xy = 0.2
        max_push_ang_vel = 0.4
        # 动作再叠加 0.02 × N(0,1) × 动作本身的乘性噪声。
        # PaiFreeEnv.step() 还用随机系数混合本步动作与上步动作以模拟延迟。
        dynamic_randomization = 0.02
        # 标准训练保持原有的随机动作延迟；阶段配置可以单独关闭或调整范围。
        randomize_action_delay = True
        action_delay_range = [0.0, 1.0]

    class commands(LeggedRobotCfg.commands):
        # 内部顺序：[机身 x 速度, 机身 y 速度, 偏航角速度, 世界系目标朝向]。
        num_commands = 4
        # 回合重置及每 8 秒重采样；很小的平面速度指令会被置零。
        resampling_time = 8.0
        # True 时从朝向误差实时计算 yaw 速度；ang_vel_yaw 范围不参与采样。
        heading_command = True

        class ranges:
            # x/y 速度单位 m/s；yaw 角速度单位 rad/s；目标朝向单位 rad。
            lin_vel_x = [-0.3, 0.6]  # min max [m/s]
            lin_vel_y = [-0.3, 0.3]  # min max [m/s]
            ang_vel_yaw = [-0.3, 0.3]  # min max [rad/s]
            heading = [-3.14, 3.14]

    class rewards:
        # 以下字段是奖励函数内部的目标值和阈值，不是各项在总奖励中的权重。
        base_height_target = 0.3453  # 机身相对参考支撑脚的目标高度，单位 m
        min_dist = 0.15  # 两脚/两膝的最小期望水平间距，单位 m
        max_dist = 0.2  # 两脚最大期望间距；两膝函数将此值乘 2，单位 m
        target_joint_pos_scale = 0.08  # 正弦参考步态的基础关节摆幅，单位 rad
        target_feet_height = 0.02  # 摆动脚累计抬高量的目标值，单位 m
        cycle_time = 0.4  # 左右交替步态的完整周期，单位 s
        # True 只把非终止项的总和截到 >=0，随后仍会单独加入终止惩罚。
        only_positive_rewards = True
        # 跟踪公式实际为 exp(-((误差/a)^2)/a)，误差先按 1+|指令| 归一化。
        tracking_sigma_ang = 0.1  # yaw 角速度跟踪的 a；越小越严格
        tracking_sigma_lin = 0.1  # x/y 线速度跟踪的 a；越小越严格
        max_contact_force = 100  # 足部接触力超过此阈值的部分被惩罚，单位 N

        class scales:
            # 每个非零名称匹配 PaiFreeEnv._reward_<名称>()；每步权重还乘策略 dt。
            # 正权重鼓励较大的函数值，负权重惩罚较大的函数值；0 表示停用。
            joint_pos = 1.6  # 鼓励 6 个髋俯仰/膝/踝俯仰角跟随正弦参考轨迹
            feet_clearance = 5.0  # 摆动脚累计抬高量距目标 <0.01 m 时给奖励
            feet_contact_number = 1.2  # 双脚实际接触与参考支撑/摆动相符时给奖励
            feet_air_time = 1.0  # 接触过滤信号出现时奖励之前累计的离地时间，上限 0.5 s
            # 名称意图是脚底防滑；实现取脚刚体状态 10:12，即 x/y 角速度而非线速度。
            foot_slip = -0.05  # 接触时惩罚上述角速度范数的平方根
            feet_distance = 0.16  # 鼓励两脚的水平距离处于 [min_dist, max_dist]
            knee_distance = 0.16  # 鼓励两膝的水平距离处于 [min_dist, 2×max_dist]
            feet_contact_forces = -0.001  # 惩罚足部接触力超过 max_contact_force 的部分
            tracking_lin_vel = 10  # 鼓励机身 x/y 速度跟踪线速度指令
            tracking_ang_vel = 20  # 鼓励机身 z 轴角速度跟踪 yaw 指令
            vel_mismatch_exp = 0.5  # 鼓励较小的竖直线速度和 x/y 角速度
            low_speed =0.05  # |x 指令|>0.1 时，根据前进速度大小/方向给奖或扣分
            track_vel_hard = 0.2  # 对线速度及 yaw 误差再加指数奖励与线性扣分
            default_hip_roll_joint_pos = 4  # 鼓励两侧髋 roll、踝 roll 关节接近零角
            default_thigh_joint_pos = 1.0  # 鼓励两侧 thigh 关节接近零角
            default_ankle_roll_pos = 0.5  # 鼓励两只脚的 roll/pitch 欧拉角接近零
            orientation = 0.5  # 鼓励机身 roll/pitch 和水平投影重力较小
            base_height = 0.5  # 鼓励机身相对参考支撑脚的高度接近目标值
            base_acc = 0.2  # 鼓励相邻步根节点线/角速度变化小；实现未除以 dt
            action_smoothness = -0.002  # 惩罚动作一阶/二阶变化及动作绝对值
            torques = -1e-5  # 惩罚 12 个关节的力矩平方和
            dof_vel = -5e-5  # 惩罚 12 个关节的角速度平方和
            dof_acc = -1e-8  # 惩罚关节角速度差除以策略 dt 后的平方和
            collision = -1.0  # 底座接触力范数超过 0.1 时计碰撞惩罚

            # 该函数对非超时终止返回 -1；终止项在其他奖励截断之后加入。
            termination = 1.0  # 因而实际产生非超时终止惩罚

    class normalization:
        class obs_scales:
            # 拼接观测前按分量乘系数；不是网络内的运行均值/方差归一化。
            # lin_vel 用于指令和 Critic 的机身线速度；Actor 不直接看到实际线速度。
            lin_vel = 2.0
            ang_vel = 1.0
            dof_pos = 1.0
            dof_vel = 0.05
            # 名称沿用 quat，但实现中缩放的是机身欧拉角。
            quat = 1.0
            # 当前不测地形高度，所以此系数未用到。
            height_measurements = 5.0

        # 送入网络的观测与送入 PD 前的动作分别裁剪到 ±18。
        clip_observations = 18.0
        clip_actions = 18.0


class PaiCfgStage0(PaiCfg):
    """阶段 0 的单一评估配置。

    阶段 0 后续需要切换摩擦、推力、噪声或测试指令时，只修改本类；
    play.py 会读取 evaluation 中的开关和用例，不再为该任务硬编码随机化。
    默认值表示名义评估：固定平地、固定摩擦、无噪声、无推力、无动作延迟。
    """

    class evaluation:
        # 默认随机种子：同一组对照保持不变；做多 seed 验证时改成其他整数。
        # 命令行 --seed 的优先级更高，可在不改文件的情况下临时覆盖。
        seed = 123145
        # 是否由 play.py 按 push_interval_s 手工施加推力。
        # 名义/摩擦/噪声单因素评估设 False；推力与组合压力评估设 True。
        # 设 True 时还必须把 max_push_vel_xy/max_push_ang_vel 改成非零值。
        scheduled_push = False
        # 是否录制每个 case 的 MP4；纯指标评估可设 False 以节省时间和显存。
        render = True
        # 是否导出 policy_torch.pt 和 policy_onnx.onnx。
        # 只做评估时可设 False；设 True 会更新共享 exported/policies 目录中的文件。
        export_policy = True
        # 固定测试用例，格式为：(名称, 机身 x 速度, 机身 y 速度, 世界系目标朝向)。
        # 速度单位 m/s，heading 单位 rad；最后一项不是直接的 yaw rate。
        # 可增删 case 或修改速度档位，但对照实验应保持同一组 cases。
        cases = (
            ("stand", 0.0, 0.0, 0.0),
            ("forward_slow", 0.3, 0.0, 0.0),
            ("forward_fast", 0.6, 0.0, 0.0),
            ("backward", -0.3, 0.0, 0.0),
            ("left", 0.0, 0.3, 0.0),
            ("right", 0.0, -0.3, 0.0),
            ("turn_left", 0.0, 0.0, 1.5707963267948966),
            ("turn_right", 0.0, 0.0, -1.5707963267948966),
            ("turn_around", 0.0, 0.0, 3.141592653589793),
            ("forward_turn_left", 0.3, 0.0, 1.5707963267948966),
            ("forward_turn_right", 0.3, 0.0, -1.5707963267948966),
        )

    class env(PaiCfg.env):
        # 并行评估环境数：RTX 2080 Ti 建议 32～64；增大可增加样本数但占用更多显存。
        # 命令行 --num_envs 仍可临时覆盖该值。
        num_envs = 64
        # 单个 case 的最长持续时间，单位 s；改变后应同步保证
        # commands.resampling_time 大于该值，避免固定指令在回合中途被重采样。
        episode_length_s = 12

    class terrain(PaiCfg.terrain):
        # Stage0 建议始终使用 "plane"；改成 heightfield/trimesh 会改变任务难度，
        # 应作为后续独立实验，不与摩擦、推力等单因素评估混在一起。
        mesh_type = "plane"
        # Stage0 保持 False；True 会随训练进度改变地形，不适合固定 checkpoint 评估。
        curriculum = False
        # Stage0 保持 False；True 会改变观测内容/维度，现有策略不能直接兼容。
        measure_heights = False
        # 地面材质摩擦。名义值保持 0.6；测试地面材料时可同时改为目标实测值。
        # 机器人碰撞体摩擦由下方 domain_rand.friction_range 单独控制。
        static_friction = 0.6
        dynamic_friction = 0.6

    class noise(PaiCfg.noise):
        # 观测噪声总开关：名义、摩擦和推力单因素测试设 False；噪声测试设 True。
        add_noise = False
        # 各观测分量噪声的总倍率；只有 add_noise=True 时生效。
        # 建议从 0.1 -> 0.3 -> 0.5 逐级测试；具体分量比例继承 PaiCfg.noise_scales。
        noise_level = 0.0

    class domain_rand(PaiCfg.domain_rand):
        # 建议保持 True，即使做固定摩擦评估也用相同上下界显式赋值，
        # 避免设 False 后机器人碰撞体沿用 URDF 中可能不同的摩擦值。
        randomize_friction = True
        # 机器人碰撞体摩擦采样范围。
        # 名义评估：[0.6, 0.6]；单点扫描可改成 [0.4, 0.4]/[0.8, 0.8]；
        # 范围压力测试可从 [0.5, 0.8] 开始，再逐步扩到 [0.4, 1.0]。
        friction_range = [0.6, 0.6]

        # base 质量随机化开关：Stage0 默认 False；只有质量单因素/组合测试才设 True。
        randomize_base_mass = False
        # 在 URDF base 质量上增加的 kg 范围，仅在 randomize_base_mass=True 时生效。
        # 建议从较小范围（如 [-0.3, 0.3]）开始，不要与新摩擦范围同时首次引入。
        added_mass_range = [0.0, 0.0]

        # play.py 已负责可复现的定时推力，因此 Stage0 应保持 False，避免自动推力叠加。
        # 是否启用评估推力只修改 evaluation.scheduled_push。
        push_robots = False
        # 手工推力之间的时间间隔，单位 s；可用 8 -> 4 -> 2 逐渐增加扰动频率。
        push_interval_s = 4
        # x/y 最大速度覆盖扰动，单位 m/s。名义评估设 0；推力测试可从 0.05、0.1
        # 逐步增至 0.2。该实现改写 base 速度，并不等同于施加真实外力。
        max_push_vel_xy = 0.0
        # x/y/z 最大角速度覆盖扰动，单位 rad/s。名义评估设 0；建议从 0.1 逐步测试。
        max_push_ang_vel = 0.0

        # 乘性动作噪声强度：action += scale * N(0,1) * action。
        # 名义评估设 0；动作噪声单因素测试建议 0.01 -> 0.02 -> 0.05。
        dynamic_randomization = 0.0
        # 动作延迟总开关：名义及其他单因素评估设 False；延迟测试才设 True。
        randomize_action_delay = False
        # 当前动作与上一帧动作的混合系数范围，仅在开关为 True 时生效。
        # 0 表示完全执行当前动作，1 表示完全执行上一帧动作；建议从 [0.0, 0.1]
        # 开始，再测试 [0.0, 0.3]/[0.0, 0.5]，不要一开始直接使用 [0.0, 1.0]。
        action_delay_range = [0.0, 0.0]

    class commands(PaiCfg.commands):
        # 指令自动重采样间隔，单位 s。Stage0 的指令由 evaluation.cases 固定写入，
        # 因此该值必须大于 env.episode_length_s；若回合改成 20 s，可相应设为 21 s。
        resampling_time = 13.0

    class sim(PaiCfg.sim):
        # PhysX GPU 最大接触对容量。数值越大越占显存，但太小会出现接触缓冲错误。
        # 64 个平地环境通常可从 2**10～2**12 测试；保留 2**23 最稳妥但更占显存。
        # 只在出现显存压力时逐级降低，并确认接触、终止和奖励结果没有变化。
        max_gpu_contact_pairs = 2**23


# 训练任务配置
class PaiCfgPPO(LeggedRobotCfgPPO):
    """标准 pai_ppo 任务的网络、PPO 超参数和训练调度。"""

    # 随机种子会传给环境；Runner 类名由 TaskRegistry 解析。
    seed = 5
    runner_class_name = "OnPolicyRunner"  # DWLOnPolicyRunner

    class policy:
        # 12 维高斯动作分布的初始标准差，以及 Actor/Critic 的隐藏层宽度。
        # 当前输入/输出为 Actor 705→512→256→128→12，Critic 219→768→256→128→1。
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [768, 256, 128]

    class algorithm(LeggedRobotCfgPPO.algorithm):
        # 只覆盖与通用 PPO 默认值不同的项；clip_param 等参数继承基类。
        # entropy_coef 控制探索，gamma/lam 分别是折扣率与 GAE 参数。
        entropy_coef = 0.001
        # Adam 初始学习率；基类 schedule='adaptive' 会按 KL 调整。
        learning_rate = 1e-5
        # 每轮采集的数据训练 2 遍，每遍分为 4 个小批量。
        num_learning_epochs = 2
        gamma = 0.994
        lam = 0.9
        num_mini_batches = 4

    class runner:
        # 类名由 OnPolicyRunner 在其模块作用域解析。
        policy_class_name = "ActorCritic"
        algorithm_class_name = "PPO"
        # 每轮样本数 = num_envs × 24；max_iterations 是更新轮数。
        num_steps_per_env = 24  # per iteration
        max_iterations = 10001  # number of policy updates

        # 日志目录为 logs/<experiment_name>/<时间>_<run_name>/；周期性存检查点。
        save_interval = 1000  # check for potential saves every this many iterations
        experiment_name = "Pai_ppo"
        run_name = "v1"
        # resume=True 时按 load_run/checkpoint 定位已有模型；-1 表示最新。
        resume = False
        load_run = -1  # -1 = last run
        checkpoint = -1  # -1 = last saved model
        resume_path = None  # updated from load_run and checkpoint


class PaiCfgStage0PPO(PaiCfgPPO):
    """Stage0 独立日志配置；网络和 PPO 参数全部继承标准基线。"""

    class runner(PaiCfgPPO.runner):
        # pai_stage0 的训练、续训和评估统一使用 logs/Pai_stage0/。
        experiment_name = "Pai_stage0"
        run_name = "stage0"


class PaiCfgMyPPO(LeggedRobotCfgPPO):
    """已有的 MyPPO 实验配置；与标准 pai_ppo 使用不同 Runner/算法。"""

    seed = 5
    # 自定义 Runner 须在 task_registry.py 的模块作用域可见。
    runner_class_name = "MyOnPolicyRunner"

    class policy:
        # 保持与 Pi 环境相同的观测/动作维度和 MLP 宽度。
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [768, 256, 128]

    class algorithm(LeggedRobotCfgPPO.algorithm):
        # 自定义算法会继承基类其余 PPO 参数；需检查 MyPPO 构造函数是否接收这些字段。
        entropy_coef = 0.001
        learning_rate = 1e-5
        num_learning_epochs = 2
        gamma = 0.994
        lam = 0.9
        num_mini_batches = 4

    class runner:
        policy_class_name = "ActorCritic"
        # MyPPO 须在 MyOnPolicyRunner 的模块作用域可见。
        algorithm_class_name = "MyPPO"
        # 此实验仅计划 10 轮更新，每轮每环境收集 24 步。
        num_steps_per_env = 24  # per iteration
        max_iterations = 10  # number of policy updates

        # 与标准 PPO 分开存放日志和检查点，避免覆盖基线结果。
        save_interval = 10  # check for potential saves every this many iterations
        experiment_name = "Pai_my_ppo"
        run_name = "v1"
        # load and resume
        resume = False
        load_run = -1  # -1 = last run
        checkpoint = -1  # -1 = last saved model
        resume_path = None  # updated from load_run and checkpoint
