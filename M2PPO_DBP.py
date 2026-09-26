# M2PPO_DP.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR


class ActorCritic(nn.Module):
    def __init__(self, L_num, input_dim=4, hidden_dim=32):
        super().__init__()
        # input_dim=4: [s_i^t, n_i^t, g_t, mu_i^t]
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.actor = nn.Linear(hidden_dim, 2)
        self.critic = nn.Linear(hidden_dim, 1)
        global_input_dim = L_num * L_num  # flatten 后的维度
        self.central_value = nn.Sequential(
            nn.Linear(global_input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        shared = self.shared(x)
        action_logits = self.actor(shared)
        action_probs = F.softmax(action_logits, dim=-1)
        state_value = self.critic(shared)
        state_value = state_value.squeeze()
        return action_probs, state_value

    # ===== MAPPO 新增：centralized critic 的前向（输入全局 L×L 状态） =====
    def forward_central_value(self, global_state):
        """
        global_state: [B, L, L] 的 0/1 策略矩阵
        返回: [B] 的 centralized value
        """
        x = global_state.float().view(global_state.shape[0], -1)  # 展成 B × (L*L)
        v = self.central_value(x)
        return v.squeeze(-1)


class SPGG_M2PPO_DBP(nn.Module):
    def __init__(self, L_num, device, alpha, gamma, clip_epsilon, r, epochs,
                 now_time, question, ppo_epochs, batch_size, gae_lambda,
                 output_path, delta, rho, p_punish):
        super().__init__()
        self.L_num = L_num
        self.device = device
        self.r = r
        self.epochs = epochs
        self.question = question
        self.now_time = now_time

        # PPO超参数
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.gae_lambda = gae_lambda
        self.delta = delta  # w_cl
        self.rho = rho  # w_ent

        # DP惩罚参数
        self.p_punish = p_punish  # density punishment strength

        self.output_path = output_path

        # 神经网络（input_dim=4：LMF状态）
        self.policy = ActorCritic(L_num=self.L_num, input_dim=4, hidden_dim=32).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=alpha)
        self.scheduler = StepLR(self.optimizer, step_size=1000, gamma=0.5)  # 每1000步学习率降低50%

        # ===== 修改：八邻域卷积核（Moore neighborhood: 周围8个邻居 + 自身）=====
        # 用于密度惩罚的八邻域核（不包括自身）
        self.moore_kernel_no_self = torch.tensor(
            [[[[1, 1, 1],
               [1, 0, 1],
               [1, 1, 1]]]],
            dtype=torch.float32, device=device
        )

        # 用于SPGG基础收益的四邻域核（von Neumann: 上、下、左、右 + 自身）
        self.von_neumann_kernel = torch.tensor(
            [[[[0, 1, 0],
               [1, 1, 1],
               [0, 1, 0]]]],
            dtype=torch.float32, device=device
        )

        # 初始化状态
        self.initial_state = self._init_state(question)
        self.current_state = self.initial_state.clone()

        # 经验缓冲区
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.global_states = []
        self.global_next_states = []

    def _init_state(self, question):
        if question == 1:  # question 1: 伯努利分布随机50%概率背叛和合作
            state = torch.bernoulli(torch.full((self.L_num, self.L_num), 0.5))
        elif question == 2:  # question 2: 上半背叛，下半合作
            state = torch.zeros(self.L_num, self.L_num)
            state[self.L_num // 2:, :] = 1
        elif question == 3:  # question 3: 全背叛
            state = torch.zeros(self.L_num, self.L_num)
        elif question == 4:
            # 创建一个 L x L 的零矩阵（棋盘格）
            state = torch.zeros((self.L_num, self.L_num))
            for i in range(self.L_num):
                for j in range(self.L_num):
                    if (i + j) % 2 == 0:
                        state[i, j] = 1
        return state.to(self.device)

    def encode_state_with_lmf(self, state_matrix):
        """
        LMF状态编码，返回 [L, L, 4] 的特征向量
        维度: [s_i^t, n_i^t, g_t, mu_i^t]
        注意：这里的 n_i^t 和 mu_i^t 使用四邻域（保持与SPGG基础模型一致）
        """
        # 添加 batch 和 channel 维度 [B, C, H, W]
        state_4d = state_matrix.float().unsqueeze(0).unsqueeze(0)  # [1, 1, L, L]

        # 使用环形填充
        padded = F.pad(state_4d, (1, 1, 1, 1), mode='circular')

        # 计算四邻域合作数 n_i^t（包括自身）- 用于LMF状态
        neighbor_coop = F.conv2d(padded, self.von_neumann_kernel).squeeze()  # [L, L]

        # 计算全局合作频率 g_t
        global_coop = torch.mean(state_matrix.float())

        # 计算局部平均场 mu_i^t = 邻居合作者数 / k（不包括自身）
        # 注意：neighbor_coop包括自身，所以需要减去自身，k=4
        mu_i = (neighbor_coop - state_matrix.float()) / 4.0

        # 构建LMF状态特征 [L, L, 4]
        lmf_state = torch.stack([
            state_matrix.float(),  # s_i^t
            neighbor_coop,  # n_i^t (包括自身)
            global_coop.expand_as(state_matrix),  # g_t
            mu_i  # mu_i^t
        ], dim=-1)

        return lmf_state.view(-1, 4)  # [L*L, 4]

    def calculate_reward_with_density_punishment(self, state_matrix):
        float_state = state_matrix.float().unsqueeze(0).unsqueeze(0)
        padded_state = F.pad(float_state, (1, 1, 1, 1), mode='circular')
        
        N_C_g = F.conv2d(padded_state, self.von_neumann_kernel)  # [1, 1, L, L]
        group_gross_profit = (self.r * N_C_g) / 5.0  # [1, 1, L, L]
        padded_gross = F.pad(group_gross_profit, (1, 1, 1, 1), mode='circular')
        # 再次卷积，把自身群体红利 + 4个邻居群体的红利完美累加
        total_gross_profit = F.conv2d(padded_gross, self.von_neumann_kernel).squeeze()  # [L, L]
        total_cost = state_matrix.float() * 5.0
        base_reward = total_gross_profit - total_cost
        padded_for_moore = F.pad(state_matrix.float().unsqueeze(0).unsqueeze(0),
                                (1, 1, 1, 1), mode='circular')
        
        neighbor_coop_moore = F.conv2d(padded_for_moore, self.moore_kernel_no_self).squeeze()
        neighbor_defector_moore = 8 - neighbor_coop_moore
        
        is_defector = (~state_matrix.bool()).float()
        
        punishment = -self.p_punish * is_defector * (neighbor_defector_moore / 8.0 + 1.0)
        total_reward = base_reward + punishment
        
        return total_reward

    def ppo_update(self):
        # 堆叠 buffer
        states = torch.stack(self.states).to(self.device)  # [T, L*L, 4]
        actions = torch.stack(self.actions).to(self.device)  # [T, L, L]
        old_log_probs = torch.stack(self.log_probs).to(self.device)  # [T, L, L]
        rewards = torch.stack(self.rewards).to(self.device)  # [T, L, L]
        next_states = torch.stack(self.next_states).to(self.device)  # [T, L*L, 4]
        dones = torch.stack(self.dones).to(self.device)  # [T, L, L]

        global_states = torch.stack(self.global_states).to(self.device)  # [T, L, L]
        global_next_states = torch.stack(self.global_next_states).to(self.device)  # [T, L, L]

        # ===== 使用 MAPPO centralized critic 计算 V(s), V(s') =====
        with torch.no_grad():
            values_scalar = self.policy.forward_central_value(global_states)  # [T]
            next_values_scalar = self.policy.forward_central_value(global_next_states)  # [T]

        # 广播到每个格点
        values = values_scalar.view(-1, 1, 1).expand_as(rewards)  # [T, L, L]
        next_values = next_values_scalar.view(-1, 1, 1).expand_as(rewards)

        # ===== GAE & advantage 计算 =====
        advantages = torch.zeros_like(rewards)
        last_advantage = torch.zeros_like(rewards[-1])

        for t in reversed(range(len(rewards))):
            dones_float = dones[t].float()
            psi = rewards[t] + self.gamma * next_values[t] * (1 - dones_float) - values[t]
            advantages[t] = psi + self.gamma * self.gae_lambda * last_advantage
            last_advantage = advantages[t]

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        returns = advantages + values  # [T, L, L]

        # ===== PPO 更新 =====
        for _ in range(self.ppo_epochs):
            for batch in self._make_batch(states, actions, old_log_probs, advantages, returns, global_states):
                state_b, action_b, old_log_b, adv_b, ret_b, gstate_b = batch
                state_b = state_b.view(-1, 4)
                action_b = action_b.reshape(-1)
                old_log_b = old_log_b.reshape(-1)
                adv_b = adv_b.reshape(-1)
                if ret_b.shape[0] == 1:
                    ret_b = ret_b.squeeze()

                # 策略仍用原来的局部特征
                probs, _ = self.policy(state_b)
                dist = Categorical(probs)
                log_probs = dist.log_prob(action_b).view_as(action_b)
                entropy = dist.entropy().mean()

                ratio = (log_probs - old_log_b).exp()
                surr1 = ratio * adv_b
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * adv_b
                actor_loss = -torch.min(surr1, surr2).mean()

                # centralized critic loss
                if ret_b.dim() == 0:
                    ret_b = ret_b.view(1, 1, 1)
                elif ret_b.dim() == 1:
                    ret_b = ret_b.view(-1, 1, 1)
                elif ret_b.dim() == 2:
                    ret_b = ret_b.unsqueeze(0)

                target_team = ret_b.mean(dim=(1, 2)).detach()  # [B]
                value_pred = self.policy.forward_central_value(gstate_b)  # [B]
                critic_loss = F.mse_loss(value_pred, target_team)

                loss = actor_loss + self.delta * critic_loss - self.rho * entropy

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()
                self.scheduler.step()

    def shot_pic(self, type_t_matrix, epoch, r, profit_data):
        """保存策略矩阵快照与数据文件（与原Q-learning代码相同格式）"""
        plt.clf()
        plt.close("all")

        # 创建输出目录
        img_dir = f'{self.output_path}/shot_pic/r={r}/two_type'
        matrix_dir = f'{self.output_path}/shot_pic/r={r}/two_type/type_t_matrix'
        profit_dir = f'{self.output_path}/shot_pic/r={r}/two_type/profit_matrix'

        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(matrix_dir, exist_ok=True)
        os.makedirs(profit_dir, exist_ok=True)

        # =============================================
        # 1. 保存策略矩阵图
        # =============================================
        fig1 = plt.figure(figsize=(8, 8))
        ax1 = fig1.add_subplot(1, 1, 1)
        ax1.axis('off')
        fig1.patch.set_edgecolor('black')
        fig1.patch.set_linewidth(2)

        color_map = {
            0: [0, 0, 0],  # 黑色（背叛）
            1: [1, 1, 1]  # 白色（合作）
        }

        strategy_image = np.zeros((self.L_num, self.L_num, 3))
        for label, color in color_map.items():
            strategy_image[type_t_matrix.cpu().numpy() == label] = color

        ax1.imshow(strategy_image, interpolation='none')
        ax1.axis('off')
        for spine in ax1.spines.values():
            spine.set_linewidth(3)

        fig1.savefig(f'{img_dir}/t={epoch}.pdf', format='pdf', dpi=300, bbox_inches='tight', pad_inches=0)
        fig1.savefig(f'{img_dir}/t={epoch}.jpg', format='jpg', dpi=300, bbox_inches='tight', pad_inches=0)

        plt.close(fig1)

        # =============================================
        # 2. 保存收益热图
        # =============================================
        if isinstance(profit_data, tuple):
            combined_reward, _, team_utility = profit_data
            profit_matrix = combined_reward
        else:
            profit_matrix = profit_data

        if not isinstance(profit_matrix, torch.Tensor):
            profit_matrix = torch.tensor(profit_matrix, device=self.device)
        profit_matrix = profit_matrix.cpu().numpy()

        fig2 = plt.figure(figsize=(8, 8))
        ax2 = fig2.add_subplot(1, 1, 1)

        vmin = 0
        vmax = np.ceil(np.maximum(5 * (r - 1), 4 * r))
        im = ax2.imshow(profit_matrix, vmin=vmin, vmax=vmax, cmap='viridis', interpolation='none')

        cbar2 = fig2.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
        cbar2.ax.tick_params(labelsize=52)

        actual_min = vmin
        actual_max = vmax
        num_ticks = 5
        tick_positions = np.linspace(actual_min, actual_max, num_ticks)  # 等距位置
        tick_labels = [str(int(round(val))) for val in tick_positions]  # 刻度标签（取整）

        cbar2.set_ticks(tick_positions)
        cbar2.set_ticklabels(tick_labels)

        ax2.set_xticks(np.arange(0, self.L_num, max(1, self.L_num // 5)))
        ax2.set_yticks(np.arange(0, self.L_num, max(1, self.L_num // 5)))
        ax2.grid(False)
        ax2.axis('off')

        fig2.savefig(f'{img_dir}/profit_t={epoch}.pdf', format='pdf', dpi=300, bbox_inches='tight', pad_inches=0)
        fig2.savefig(f'{img_dir}/profit_t={epoch}.jpg', format='jpg', dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close(fig2)
        np.savetxt(f'{matrix_dir}/T{epoch}.txt',
                   type_t_matrix.cpu().numpy(), fmt='%d')
        np.savetxt(f'{profit_dir}/T{epoch}.txt',
                   profit_matrix, fmt='%.4f')
        return 0
    def _make_batch(self, states, actions, old_log_probs, advantages, returns, global_states):
        perm = torch.randperm(len(states))
        for i in range(0, len(states), self.batch_size):
            idx = perm[i:i + self.batch_size]
            yield (
                states[idx],  # 保持 [B, L, L, 3]
                actions[idx],  # 保持 [B, L, L]
                old_log_probs[idx],  # 保持 [B, L, L]
                advantages[idx],  # 保持 [B, L, L]
                returns[idx],  # 保持 [B, L, L] ← 不要 reshape！
                global_states[idx]  # [B, L, L]
            )

    def run(self):
        coop_rates = []
        defect_rates = []
        total_values = []

        for epoch in tqdm(range(self.epochs)):
            self.epoch = epoch
            if epoch == 0:
                profit_matrix = self.calculate_reward_with_density_punishment(self.current_state)
                self.shot_pic(self.current_state, epoch, self.r, profit_matrix)
                # 记录初始状态
                coop_rate = self.current_state.float().mean().item()
                defect_rate = 1 - coop_rate
                total_value = profit_matrix.sum().item()
                coop_rates.append(coop_rate)
                defect_rates.append(defect_rate)
                total_values.append(total_value)

            # 选择动作并执行
            action, log_prob = self.choose_action(self.current_state)
            next_state = action
            reward = self.calculate_reward_with_density_punishment(next_state)
            done = torch.zeros_like(next_state, dtype=torch.bool)

            # 存储LMF状态（4维特征）
            self.states.append(self.encode_state_with_lmf(self.current_state).view(self.L_num, self.L_num, 4))
            self.actions.append(action)
            self.log_probs.append(log_prob)
            self.rewards.append(reward)
            self.next_states.append(self.encode_state_with_lmf(next_state).view(self.L_num, self.L_num, 4))
            self.dones.append(done)

            # 存储全局状态
            self.global_states.append(self.current_state.detach().cpu())
            self.global_next_states.append(next_state.detach().cpu())

            if len(self.states) >= self.batch_size * self.ppo_epochs:
                self.ppo_update()
                self.current_state = next_state
                self._reset_buffer()
            else:
                self.current_state = next_state

            if (epoch + 1 in [1, 10, 100, 1000, 10000, 100000]):
                profit_matrix = self.calculate_reward_with_density_punishment(self.current_state)
                self.shot_pic(self.current_state, epoch + 1, self.r, profit_matrix)

            if epoch % 1000 == 0:
                self.save_checkpoint()

            # 记录更新后的状态
            if epoch < self.epochs - 1:
                coop_rate = self.current_state.float().mean().item()
                defect_rate = 1 - coop_rate
                total_value = reward.sum().item()
                coop_rates.append(coop_rate)
                defect_rates.append(defect_rate)
                total_values.append(total_value)

        # 记录最终状态
        coop_rate = self.current_state.float().mean().item()
        defect_rate = 1 - coop_rate
        total_value = self.calculate_reward_with_density_punishment(self.current_state).sum().item()
        coop_rates.append(coop_rate)
        defect_rates.append(defect_rate)
        total_values.append(total_value)

        self.save_checkpoint(is_final=True)

        return defect_rates, coop_rates, [], [], total_values

    def save_data(self, data_type, name, r, data):
        output_dir = f'{self.output_path}/{data_type}'
        os.makedirs(output_dir, exist_ok=True)
        np.savetxt(f'{output_dir}/{name}.txt', data)

    def _reset_buffer(self):
        """显式释放显存"""
        del self.states[:]
        del self.actions[:]
        del self.log_probs[:]
        del self.rewards[:]
        del self.next_states[:]
        del self.dones[:]
        del self.global_states[:]
        del self.global_next_states[:]
        torch.cuda.empty_cache()

    def _store_transition(self, state, action, log_prob, reward, next_state, done):
        """存储时分离梯度"""
        self.states.append(state.detach().cpu())
        self.actions.append(action.detach().cpu())
        self.log_probs.append(log_prob.detach().cpu())
        self.rewards.append(reward.detach().cpu())
        self.next_states.append(next_state.detach().cpu())
        self.dones.append(done.detach().cpu())

    def save_checkpoint(self, is_final=False):
        """保存模型检查点"""
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'r': self.r,
            'gamma': self.gamma,
            'clip_epsilon': self.clip_epsilon,
            'p_punish': self.p_punish,
        }
        model_dir = f"{self.output_path}/checkpoint"
        os.makedirs(model_dir, exist_ok=True)
        filename = f"model_r{self.r}_final.pth" if is_final else f"model_r{self.r}_epoch{self.epoch}.pth"
        torch.save(checkpoint, f"{model_dir}/{filename}")

    def choose_action(self, state_matrix):
        with torch.no_grad():
            features = self.encode_state_with_lmf(state_matrix)
            probs, _ = self.policy(features)
            dist = Categorical(probs)
            actions = dist.sample()
        return actions.view_as(state_matrix), dist.log_prob(actions).view_as(state_matrix)