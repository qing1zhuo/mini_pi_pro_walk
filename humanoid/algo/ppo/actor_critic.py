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

import copy
import torch
import torch.nn as nn
from torch.distributions import Normal

class ActorCritic(nn.Module):
    def __init__(self,  num_actor_obs,
                        num_critic_obs,
                        num_actions,
                        actor_hidden_dims=[256, 256, 256],
                        critic_hidden_dims=[256, 256, 256],
                        frame_stack=15,
                        num_single_obs=47,
                        transformer_dim=128,
                        transformer_heads=4,
                        transformer_layers=2,
                        transformer_ff_dim=512,
                        transformer_dropout=0.0,
                        init_noise_std=1.0,
                        activation = nn.ELU(),
                        **kwargs):
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super(ActorCritic, self).__init__()

        if num_actor_obs != frame_stack * num_single_obs:
            raise ValueError(
                f"Actor observation dimension {num_actor_obs} does not match "
                f"frame_stack * num_single_obs ({frame_stack} * {num_single_obs})."
            )
        if transformer_dim % transformer_heads != 0:
            raise ValueError(
                f"transformer_dim ({transformer_dim}) must be divisible by "
                f"transformer_heads ({transformer_heads})."
            )
        if transformer_layers < 1:
            raise ValueError("transformer_layers must be at least 1.")
        if transformer_dropout != 0.0:
            raise ValueError("transformer_dropout must be 0.0 for PPO training.")
        if not actor_hidden_dims:
            raise ValueError("actor_hidden_dims must contain at least one hidden layer.")

        self.num_actor_obs = num_actor_obs
        self.frame_stack = frame_stack
        self.num_single_obs = num_single_obs
        self.transformer_dim = transformer_dim
        self.transformer_heads = transformer_heads
        self.transformer_head_dim = transformer_dim // transformer_heads
        self.attention_scale = self.transformer_head_dim ** -0.5

        mlp_input_dim_c = num_critic_obs

        # Policy: reshape the flattened history and encode it as a temporal sequence.
        frame_encoder = nn.Sequential(
            nn.Linear(num_single_obs, transformer_dim),
            nn.LayerNorm(transformer_dim),
        )
        position_embedding = nn.Embedding(frame_stack, transformer_dim)
        nn.init.normal_(position_embedding.weight, mean=0.0, std=0.02)

        transformer_blocks = nn.ModuleList()
        for _ in range(transformer_layers):
            transformer_blocks.append(
                nn.ModuleDict(
                    {
                        "norm1": nn.LayerNorm(transformer_dim),
                        "qkv": nn.Linear(transformer_dim, 3 * transformer_dim),
                        "attention_output": nn.Linear(transformer_dim, transformer_dim),
                        "norm2": nn.LayerNorm(transformer_dim),
                        "ffn": nn.Sequential(
                            nn.Linear(transformer_dim, transformer_ff_dim),
                            nn.ELU(),
                            nn.Linear(transformer_ff_dim, transformer_dim),
                        ),
                    }
                )
            )

        actor_layers = []
        actor_input_dim = 2 * transformer_dim
        actor_layers.append(nn.Linear(actor_input_dim, actor_hidden_dims[0]))
        actor_layers.append(copy.deepcopy(activation))
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                actor_layers.append(copy.deepcopy(activation))

        # Keep all Actor parameters under the existing ``actor`` attribute while
        # implementing the non-sequential data flow in _actor_forward().
        self.actor = nn.ModuleDict(
            {
                "frame_encoder": frame_encoder,
                "position_embedding": position_embedding,
                "transformer_blocks": transformer_blocks,
                "output_head": nn.Sequential(*actor_layers),
            }
        )

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        print(f"Actor Transformer: {self.actor}")
        print(f"Critic MLP: {self.critic}")

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False
        

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]


    def reset(self, dones=None):
        pass

    def forward(self, observations):
        return self._actor_forward(observations)
    
    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def _actor_forward(self, observations):
        batch_size = observations.shape[0]
        frames = observations.reshape(
            batch_size,
            self.frame_stack,
            self.num_single_obs,
        )

        frame_embeddings = self.actor["frame_encoder"](frames)
        position_embeddings = self.actor["position_embedding"].weight.unsqueeze(0)
        x = frame_embeddings + position_embeddings

        for block in self.actor["transformer_blocks"]:
            normalized_x = block["norm1"](x)
            query, key, value = torch.chunk(block["qkv"](normalized_x), 3, dim=-1)

            query = query.reshape(
                batch_size,
                self.frame_stack,
                self.transformer_heads,
                self.transformer_head_dim,
            ).transpose(1, 2)
            key = key.reshape(
                batch_size,
                self.frame_stack,
                self.transformer_heads,
                self.transformer_head_dim,
            ).transpose(1, 2)
            value = value.reshape(
                batch_size,
                self.frame_stack,
                self.transformer_heads,
                self.transformer_head_dim,
            ).transpose(1, 2)

            attention_scores = torch.matmul(query, key.transpose(-2, -1))
            attention_scores = attention_scores * self.attention_scale
            attention_weights = torch.softmax(attention_scores, dim=-1)
            attention_output = torch.matmul(attention_weights, value)
            attention_output = attention_output.transpose(1, 2).contiguous().reshape(
                batch_size,
                self.frame_stack,
                self.transformer_dim,
            )

            x = x + block["attention_output"](attention_output)
            x = x + block["ffn"](block["norm2"](x))

        temporal_feature = x[:, -1, :]
        current_feature = frame_embeddings[:, -1, :]
        actor_feature = torch.cat((temporal_feature, current_feature), dim=-1)
        return self.actor["output_head"](actor_feature)

    def update_distribution(self, observations):
        mean = self._actor_forward(observations)
        self.distribution = Normal(mean, mean*0. + self.std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        actions_mean = self._actor_forward(observations)
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        value = self.critic(critic_observations)
        return value
