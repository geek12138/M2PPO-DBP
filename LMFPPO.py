import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
from tqdm import tqdm
import asyncio
from torch.optim.lr_scheduler import StepLR

class MeanFieldActorCritic(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=64):  # 输入维度改为4，包含mean field
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.actor = nn.Linear(hidden_dim, 2)
        self.critic = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        shared = self.shared(x)
        action_logits = self.actor(shared)
        action_probs = F.softmax(action_logits, dim=-1)
        state_value = self.critic(shared)
        state_value = state_value.squeeze()
        return action_probs, state_value

class MFPPO_SPGG(nn.Module):
    def __init__(self, L_num, device, alpha, gamma, clip_epsilon, r, epochs, 
                    now_time, question, ppo_epochs, batch_size, gae_lambda,
                    output_path, delta, rho, punishment_strength=0.1):  # 新增惩罚强度参数
        super().__init__()
        self.L_num = L_num
        self.device = device
        self.r = r
        self.epochs = epochs
        self.question = question
        self.now_time = now_time
        
        # PPO超参数（保持不变）
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.gae_lambda = gae_lambda
        self.delta = delta  # w_cl
        self.rho = rho  # w_ent
        
        # 新增：惩罚强度参数
        self.punishment_strength = punishment_strength

        self.output_path = output_path
        
        # 神经网络 - 使用Mean Field网络
        self.policy = MeanFieldActorCritic().to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=alpha)
        self.scheduler = StepLR(self.optimizer, step_size=1000, gamma=0.5)
        
        # 邻域卷积核
        self.neibor_kernel = torch.tensor(
            [[[[0,1,0], [1,1,1], [0,1,0]]]], 
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
        
        # Mean Field相关变量
        self.mean_actions = []  # 存储每个时间步的平均动作

    def _init_state(self, question):
        if question == 1:  # 伯努利分布随机50%概率背叛和合作
            state = torch.bernoulli(torch.full((self.L_num, self.L_num), 0.5))
        elif question == 2:  # 上半背叛，下半合作
            state = torch.zeros(self.L_num, self.L_num)
            state[self.L_num//2:, :] = 1
        elif question == 3:  # 全背叛
            state = torch.zeros(self.L_num, self.L_num)
        elif question == 4:  # 交替模式
            state = torch.zeros((self.L_num, self.L_num))
            for i in range(self.L_num):
                for j in range(self.L_num):
                    if (i + j) % 2 == 0:
                        state[i, j] = 1
        return state.to(self.device)

    def compute_mean_action(self, state_matrix):
        """计算每个智能体的邻居平均动作（Mean Field核心）"""
        # 使用卷积计算每个位置的邻居合作者数量
        padded_state = F.pad(state_matrix.float().unsqueeze(0).unsqueeze(0), 
                           (1, 1, 1, 1), mode='circular')
        neighbor_coop = F.conv2d(padded_state, self.neibor_kernel).squeeze()
        
        # 计算平均动作（邻居合作者比例）
        mean_action = neighbor_coop / 4.0  # 4邻居
        return mean_action

    def encode_state(self, state_matrix, mean_action_matrix):
        """将2D网格转换为包含Mean Field信息的特征"""
        # 计算全局合作比例
        global_coop = torch.mean(state_matrix.float())
        
        # 计算邻居合作者数量
        padded_state = F.pad(state_matrix.float().unsqueeze(0).unsqueeze(0), 
                           (1, 1, 1, 1), mode='circular')
        neighbor_coop = F.conv2d(padded_state, self.neibor_kernel).squeeze()
        
        # 构建包含Mean Field的特征
        return torch.stack([
            state_matrix.float().squeeze(),           # 自身策略
            neighbor_coop,                           # 邻居合作者数量
            global_coop.expand_as(state_matrix),     # 全局合作比例
            mean_action_matrix                       # Mean Field: 邻居平均动作
        ], dim=-1).view(-1, 4)  # 输入维度为4

    def calculate_punishment(self, state_matrix):
        """计算基于不平衡理论的惩罚收益（无成本版本）"""
        # 1. 对状态矩阵进行padding处理（环形边界）
        padded_state = F.pad(state_matrix.float().unsqueeze(0).unsqueeze(0), 
                        (1, 1, 1, 1), mode='circular')
        
        # 2. 计算每个位置的邻居背叛者数量
        neighbor_defect = F.conv2d((1 - padded_state), self.neibor_kernel).squeeze()
        
        # 3. 计算惩罚矩阵：只有合作者才能惩罚背叛者
        punishment_matrix = torch.zeros_like(state_matrix, dtype=torch.float32)
        
        # 合作者位置（state_matrix == 1）
        cooperator_positions = (state_matrix == 1)
        
        # 4. 背叛者受到惩罚（从合作者那里），但合作者不承担成本
        # 背叛者受到的惩罚 = -惩罚强度 × 邻居合作者数量
        # 注意：这里惩罚的是背叛者，不是合作者付出成本
        
        # 计算每个位置的邻居合作者数量
        neighbor_coop = F.conv2d(padded_state, self.neibor_kernel).squeeze()
        
        # 背叛者位置（state_matrix == 0）
        defector_positions = (state_matrix == 0)
        
        # 背叛者受到的惩罚：与邻居合作者数量成正比
        punishment_matrix[defector_positions] = -self.punishment_strength * neighbor_coop[defector_positions]
        
        return punishment_matrix

    def calculate_reward_with_punishment(self, state_matrix):
        """计算包含惩罚机制的收益（无成本版本）"""
        # 1. 计算基础收益（原有逻辑）
        base_reward = self.calculate_reward(state_matrix)
        
        # 2. 计算惩罚收益（背叛者受到惩罚，合作者不付出成本）
        punishment_reward = self.calculate_punishment(state_matrix)
        
        # 3. 合并收益
        total_reward = base_reward + punishment_reward
        
        return total_reward

    def calculate_reward(self, state_matrix):
        """计算每个智能体参与的5组博弈的总收益（基础收益，不含惩罚）"""
        # 1. 对状态矩阵进行padding处理（环形边界）
        padded = F.pad(state_matrix.float().unsqueeze(0).unsqueeze(0), (1,1,1,1), mode='circular')
        
        # 2. 计算每个位置的邻域合作者数量（4邻居）
        neighbor_coop = F.conv2d(padded, self.neibor_kernel).squeeze()
        
        # 3. 计算中心智能体作为合作者时的单组收益 (r*n_C/5 - 1)
        c_single_profit = (self.r * neighbor_coop / 5) - 1
        
        # 4. 计算中心智能体作为背叛者时的单组收益 (r*n_C/5)
        d_single_profit = (self.r * neighbor_coop / 5)
        
        # 5. 对单组收益矩阵进行padding处理
        padded_c_profit = F.pad(c_single_profit.unsqueeze(0).unsqueeze(0), (1,1,1,1), mode='circular')
        padded_d_profit = F.pad(d_single_profit.unsqueeze(0).unsqueeze(0), (1,1,1,1), mode='circular')
        
        # 6. 计算每个智能体参与的5组博弈总收益
        c_total_profit = F.conv2d(padded_c_profit, self.neibor_kernel).squeeze()
        d_total_profit = F.conv2d(padded_d_profit, self.neibor_kernel).squeeze()
        
        # 7. 根据当前策略选择对应的总收益
        reward_matrix = torch.where(state_matrix.bool(), c_total_profit, d_total_profit)
        
        return reward_matrix

    def ppo_update(self):
        """PPO更新（保持不变）"""
        states = torch.stack(self.states)
        actions = torch.stack(self.actions)
        old_log_probs = torch.stack(self.log_probs)
        rewards = torch.stack(self.rewards)
        next_states = torch.stack(self.next_states)
        dones = torch.stack(self.dones)

        with torch.no_grad():
            _, values = self.policy(states)
            _, next_values = self.policy(next_states)

        advantages = torch.zeros_like(rewards)
        last_advantage = 0
        for t in reversed(range(len(rewards))):
            dones_float = dones[t].float()
            psi = rewards[t] + self.gamma * next_values[t] * (1 - dones_float) - values[t]
            advantages[t] = psi + self.gamma * self.gae_lambda * last_advantage
            last_advantage = advantages[t]

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        returns = advantages + values

        for _ in range(self.ppo_epochs):
            for batch in self._make_batch(states, actions, old_log_probs, advantages, returns):
                state_b, action_b, old_log_b, adv_b, ret_b = batch
                if ret_b.shape[0] == 1:
                    ret_b = ret_b.squeeze()
                probs, value_pred = self.policy(state_b)
                dist = Categorical(probs)
                log_probs = dist.log_prob(action_b).view_as(action_b)
                entropy = dist.entropy().mean()
                
                ratio = (log_probs - old_log_b).exp()
                surr1 = ratio * adv_b
                surr2 = torch.clamp(ratio, 1-self.clip_epsilon, 1+self.clip_epsilon) * adv_b
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = F.mse_loss(value_pred, ret_b)

                loss = actor_loss + self.delta * critic_loss - self.rho * entropy
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()
                self.scheduler.step()

    def _make_batch(self, states, actions, old_log_probs, advantages, returns):
        """创建批次数据"""
        perm = torch.randperm(len(states))
        for i in range(0, len(states), self.batch_size):
            idx = perm[i:i+self.batch_size]
            yield (states[idx], actions[idx], old_log_probs[idx], advantages[idx], returns[idx])

    def choose_action(self, state_matrix):
        """选择动作（包含Mean Field信息）"""
        with torch.no_grad():
            # 计算当前状态的Mean Field
            mean_action_matrix = self.compute_mean_action(state_matrix)
            # 编码状态（包含Mean Field）
            features = self.encode_state(state_matrix, mean_action_matrix)
            probs, _ = self.policy(features)
            dist = Categorical(probs)
            actions = dist.sample()
            log_probs = dist.log_prob(actions)
            
        return actions.view_as(state_matrix), log_probs.view_as(state_matrix), mean_action_matrix

    def run(self):
        """主运行循环"""
        coop_rates = []
        defect_rates = []
        total_values = []
        punishment_values = []  # 新增：记录惩罚值
        
        for epoch in tqdm(range(self.epochs)):
            self.epoch = epoch
            
            # 选择动作（返回动作、对数概率和Mean Field）
            action, log_prob, mean_action = self.choose_action(self.current_state)
            
            # 执行环境步骤
            next_state = action
            # 使用包含惩罚的收益计算
            reward = self.calculate_reward_with_punishment(next_state)
            done = torch.zeros_like(next_state, dtype=torch.bool)
            
            # 计算惩罚值用于记录
            punishment = self.calculate_punishment(next_state)
            avg_punishment = punishment.mean().item()
            punishment_values.append(avg_punishment)
            
            # 计算下一个状态的Mean Field
            next_mean_action = self.compute_mean_action(next_state)
            
            # 存储经验（包含Mean Field信息）
            current_features = self.encode_state(self.current_state, mean_action).view(self.L_num, self.L_num, 4)
            next_features = self.encode_state(next_state, next_mean_action).view(self.L_num, self.L_num, 4)
            
            self.states.append(current_features)
            self.actions.append(action)
            self.log_probs.append(log_prob)
            self.rewards.append(reward)
            self.next_states.append(next_features)
            self.dones.append(done)
            self.mean_actions.append(mean_action)  # 存储Mean Field信息
            
            if epoch == 0:
                profit_matrix = self.calculate_reward(self.current_state)
                punishment_matrix = self.calculate_punishment(self.current_state)
                asyncio.create_task(self.shot_pic_with_punishment(
                    self.current_state, epoch, self.r, profit_matrix, punishment_matrix))
                coop_rate = self.current_state.float().mean().item()
                defect_rate = 1 - coop_rate
                total_value = reward.sum().item()
                
                coop_rates.append(coop_rate)
                defect_rates.append(defect_rate)
                total_values.append(total_value)
                
            if len(self.states) >= self.batch_size * self.ppo_epochs:
                self.ppo_update()
                self.current_state = next_state
                self._reset_buffer()
            else:
                self.current_state = next_state

            # 在关键时间点保存快照（包含惩罚信息）
            if (epoch+1 in [1, 10, 100, 1000, 10000, 100000]):
                profit_matrix = self.calculate_reward(self.current_state)
                punishment_matrix = self.calculate_punishment(self.current_state)
                asyncio.create_task(self.shot_pic_with_punishment(
                    self.current_state, epoch+1, self.r, profit_matrix, punishment_matrix))

            if epoch % 1000 == 0:
                self.save_checkpoint()
                
            # 收集数据
            coop_rate = self.current_state.float().mean().item()
            defect_rate = 1 - coop_rate
            total_value = reward.sum().item()
            
            coop_rates.append(coop_rate)
            defect_rates.append(defect_rate)
            total_values.append(total_value)
        
        # 保存惩罚数据
        self.save_data('Punishment', f'r{self.r}', self.r, punishment_values)
        
        self.save_checkpoint(is_final=True)
        return defect_rates, coop_rates, [], [], total_values

    def save_data(self, data_type, name, r, data):
        """保存数据"""
        output_dir = f'{self.output_path}/{data_type}'
        os.makedirs(output_dir, exist_ok=True)
        np.savetxt(f'{output_dir}/{name}.txt', data)

    async def shot_pic_with_punishment(self, type_t_matrix, epoch, r, profit_matrix, punishment_matrix):
        """绘制包含惩罚信息的快照"""
        plt.clf()
        plt.close("all")
        
        img_dir = f'{self.output_path}/shot_pic/r={r}/two_type'
        matrix_dir = f'{self.output_path}/shot_pic/r={r}/two_type/type_t_matrix'
        profit_dir = f'{self.output_path}/shot_pic/r={r}/two_type/profit_matrix'
        punishment_dir = f'{self.output_path}/shot_pic/r={r}/two_type/punishment_matrix'
        
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(matrix_dir, exist_ok=True)  
        os.makedirs(profit_dir, exist_ok=True)
        os.makedirs(punishment_dir, exist_ok=True)

        # 计算统计信息
        coop_rate = type_t_matrix.float().mean().item()
        defect_rate = 1 - coop_rate
        avg_profit = profit_matrix.mean().item()
        std_profit = profit_matrix.std().item()
        avg_punishment = punishment_matrix.mean().item()
        
        fig_dpi = 300

        # 子图1: 策略分布
        plt.figure(figsize=(10, 8))
        color_map = {
            0: [0.8, 0.2, 0.2],  # 红色表示背叛者
            1: [0.2, 0.6, 0.8]   # 蓝色表示合作者
        }

        strategy_image = np.zeros((self.L_num, self.L_num, 3))
        for label, color in color_map.items():
            strategy_image[type_t_matrix.cpu().numpy() == label] = color

        plt.imshow(strategy_image, interpolation='none', aspect='equal')
        plt.axis('off')
        
        from matplotlib.patches import Patch
        

        plt.tight_layout()
        
        # 同时保存 PDF 和 PNG
        pdf_path = f'{img_dir}/strategy_distribution_t={epoch}.pdf'
        png_path = f'{img_dir}/strategy_distribution_t={epoch}.png'
        plt.savefig(pdf_path, format='pdf', dpi=fig_dpi, bbox_inches='tight', pad_inches=0.1)
        plt.savefig(png_path, format='png', dpi=fig_dpi, bbox_inches='tight', pad_inches=0.1)
        plt.close()

        # 子图2: 收益分布
        plt.figure(figsize=(8, 8))
        profit_data = profit_matrix.cpu().numpy()
        vmin, vmax = 0, 9  # 颜色范围保持0-9
        plt.imshow(profit_data, cmap='viridis', vmin=vmin, vmax=vmax, interpolation='none', aspect='equal')
        plt.axis('off')
        cbar = plt.colorbar(fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=28)
        cbar.set_ticks(np.arange(0, 9, 1))  # 只显示0-8的刻度

        plt.tight_layout()
        
        # 同时保存 PDF 和 PNG
        pdf_path = f'{img_dir}/profit_t={epoch}.pdf'
        png_path = f'{img_dir}/profit_t={epoch}.png'
        plt.savefig(pdf_path, format='pdf', dpi=fig_dpi, bbox_inches='tight', pad_inches=0)
        plt.savefig(png_path, format='png', dpi=fig_dpi, bbox_inches='tight', pad_inches=0)
        plt.close()

        # 子图3: 惩罚分布
        plt.figure(figsize=(8, 8))
        punishment_data = punishment_matrix.cpu().numpy()
        punishment_abs = np.abs(punishment_data)  # 取绝对值显示惩罚强度
        vmin, vmax = 0, 4
        plt.imshow(punishment_abs, cmap='Reds', vmin=vmin, vmax=vmax, interpolation='none', aspect='equal')
        plt.axis('off')
        cbar = plt.colorbar(fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=28)
        cbar.set_ticks(np.arange(0, 4, 1))
        plt.tight_layout()
        
        # 同时保存 PDF 和 PNG
        pdf_path = f'{img_dir}/punishment_t={epoch}.pdf'
        png_path = f'{img_dir}/punishment_t={epoch}.png'
        plt.savefig(pdf_path, format='pdf', dpi=fig_dpi, bbox_inches='tight', pad_inches=0)
        plt.savefig(png_path, format='png', dpi=fig_dpi, bbox_inches='tight', pad_inches=0)
        plt.close()

    def _reset_buffer(self):
        """清空缓冲区"""
        del self.states[:]
        del self.actions[:]
        del self.log_probs[:]
        del self.rewards[:]
        del self.next_states[:]
        del self.dones[:]
        del self.mean_actions[:]
        torch.cuda.empty_cache()

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
            'punishment_strength': self.punishment_strength,  # 保存惩罚强度
        }
        model_dir = f"{self.output_path}/checkpoint"
        os.makedirs(model_dir, exist_ok=True)
        filename = f"model_r{self.r}_final.pth" if is_final else f"model_r{self.r}_epoch{self.epoch}.pth"
        torch.save(checkpoint, f"{model_dir}/{filename}")