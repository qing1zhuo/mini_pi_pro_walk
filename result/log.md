# main
1. baseline1: 
    - 主要改动：将原版仓库控制频率调整为100Hz
    - 模型：baseline1.onnx
    - 表现：
        - 前进存在后倾，后退较慢
        - 没有左移，右移极慢
2. baseline2: 
    - 主要改动：将原版仓库控制频率调整为100Hz, 并修改pai_env.py中_neg_exp&_neg_sqrd_exp两个函数
    - 模型：baseline2.onnx
    - 表现：
        - 站立时会右移
        - 前进存在向右偏航，后退很慢
        - 没有左移，右移正常
        - 左转正常，右转时无法原地转向，伴有前进
## main-mlp
1. mlp1:
    - 主要改动：baseline2基础上，加深mlp，并添加layernorm和res-connection
    ```markdown
    输入 [B, 705]
      │
    Linear 705→512 → LayerNorm → ELU
      │
     h1 ─────────────────────────┐
      │                          │
    LayerNorm → ELU → Linear 512→512  │
      │                          │
      + ◄────────────────────────┘
      │
    Linear 512→256 → LayerNorm → ELU
      │
     h3 ─────────────────────────┐
      │                          │
    LayerNorm → ELU → Linear 256→256  │
      │                          │
      + ◄────────────────────────┘
      │
    Linear 256→128 → LayerNorm → ELU
      │
    Linear 128→12
      │
    动作均值 μ [B, 12]
    ```
    - 模型：mlp1.onnx
    - 真机表现：依托
## main-net
1. net1: 
    - 主要改动：baseline1 基础上，采用新模型架构
    ```markdown
    历史观测 [B, 705]
        │
        │ reshape，按最旧帧 → 最新帧排列
        ▼
    帧序列 [B, 15, 47]
        │
        │ 共享帧编码器
        │ Linear(47,128) + LayerNorm
        ▼
    帧特征 [B, 15, 128]
        │
        │ 加可学习位置编码 [1,15,128]
        ▼
    2 × Pre-LN Transformer Block
        │
        ├─ 4-head Self-Attention
        ├─ FFN：128 → 512 → 128
        ├─ 两处残差连接
        └─ Dropout = 0
        │
        ├─ 最新时序特征 x[:, -1, :]          [B,128]
        └─ 最新原始帧特征 embedding[:, -1, :] [B,128]
                            │
                            ▼
                        拼接 [B,256]
                            │
                            ▼
                动作头：256 → 256 → 128 → 12
                            │
                            ▼
                    12维动作均值
    ```
    - 模型：net1.onnx
    - 表现：
        - 站立时存在轻微前进
        - 前进有向左偏航问题，后退很慢
        - 左移存在朝向右偏问题，右移较慢
        - 左转右转正常
        - 腿部存在外摆问题，且抬腿幅度较小
2. net2: 
    - 主要改动：baseline2 基础上，采用net1架构
    - 模型：net2.onnx
    - 表现：

3. net3: 目前最佳！！！
    - 主要改动：net2 基础上，将transformer_block加深为4层
    - 模型：net3.onnx
    - 表现：
        - 站立时仍有轻微前进，且朝向会出现左转
        - 前进已无明显偏航，但后退较慢，且后退时朝向会左转
        - 左移较慢且存在朝向右转，右移速度正常但存在朝向左转
        - 左右转正常

### main-net-cmd_sample
1. forward_fast:
    - 主要改动：net2基础上，提高高速前进采样概率。30% 概率将 vx 从 [0.45, 0.6] m/s 均匀采样，其余沿用原范围均匀采样。
    - 模型：forward_fast.onnx
    - 真机表现：
        - 站立时存在前进
        - 前进时会摔倒。后退变右移了

### main-net-reward
1. feet_h_3:
    - 主要改动：net2基础上，将目标抬脚高度提升为3cm
    - 模型：feet_h_3.onnx
    - 真机表现：起步就摔
2. fault_hip_roll_joint_pos_6:
    - 主要改动：net2基础上，default_hip_roll_joint_pos由4->6，并撤回抬脚高度的修改
    - 模型：fault_hip_roll_joint_pos_6.onnx
    - 真机表现：
        - 前进出现向右偏航，后退极慢
        - 转向极慢
3. track_ang_vel_30：
    - 主要改动：net2基础上，将角速度跟踪scale调大为30
    - 模型：track_ang_vel_30.onnx
    - 真机表现：
        - 站立存在前进问题
        - 前进正常，但后退极慢且存在右移
        - 左移无响应，右移较慢
        - 左右转正常
        - 朝向偏转问题明显改善，但总体存在运动能力和指令遵循能力减弱
4. track_ang_vel_30_2:
    - 主要改动：net3基础上，将角速度跟踪scale调大为30
    - 真机表现
        - 站立存在前进和右转
        - 前进存在右转，后退极慢且存在右转
        - 左右移较慢且均存在右转
        - 左右转正常
5. track_ang_vel_30_3:
    - 主要改动：基于net3权重，将角速度跟踪scale调大为30，续训2000轮
    - 真机表现：
        