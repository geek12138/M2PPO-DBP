import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import asyncio

class SPGG(nn.Module):
    def __init__(self, L_num, device, r, epoches, 
                    now_time, question, output_path):
        super().__init__()
        self.L_num = L_num
        self.device = device
        self.r = r
        self.epoches = epoches
        self.question = question
        self.now_time = now_time
        self.output_path=output_path
        
        # 邻域卷积核
        self.neibor_kernel = torch.tensor(
            [[[[0,1,0], [1,1,1], [0,1,0]]]], 
            dtype=torch.float32, device=device
        )
        
        # 初始化状态
        self.initial_state = self._init_state(question)
        self.current_state = self.initial_state.clone()


    def _init_state(self, question):
        if question == 1: # question 1: 伯努利分布随机50%概率背叛和合作
            state = torch.bernoulli(torch.full((self.L_num, self.L_num), 0.5))
        elif question == 2: # question 2: 上半背叛，下半合作
            state = torch.zeros(self.L_num, self.L_num)
            state[self.L_num//2:, :] = 1
        elif question == 3:  # question 3: 全背叛
            state = torch.zeros(self.L_num, self.L_num)
        elif question == 4:
            # 创建一个 L x L 的零矩阵
            state = torch.zeros((self.L_num, self.L_num))
            # 填充交替的 0 和 1
            for i in range(self.L_num):
                for j in range(self.L_num):
                    if (i + j) % 2 == 0:
                        state[i, j] = 1
        return state.to(self.device)

    def encode_state(self, state_matrix):
        """将 2D 网格转换为 4D 张量后填充"""
        # 添加 batch 和 channel 维度 [B, C, H, W]
        state_4d = state_matrix.float().unsqueeze(0).unsqueeze(0)  # [1, 1, L, L]
        
        # 使用正确的填充参数格式 (padding_left, padding_right, padding_top, padding_bottom)
        padded = F.pad(state_4d, (1, 1, 1, 1), mode='circular')  # 四周各填充1
        
        # 计算邻域合作数
        neighbor_coop = F.conv2d(padded, self.neibor_kernel).squeeze()  # [L, L]
        global_coop = torch.mean(state_matrix.float())
        return torch.stack([
            state_matrix.float().squeeze(),
            neighbor_coop,
            global_coop.expand_as(state_matrix)
        ], dim=-1).view(-1, 3)

    def calculate_reward(self, state_matrix):
        """计算每个智能体参与的5组博弈的总收益"""
        # 1. 对状态矩阵进行padding处理（环形边界）
        padded = F.pad(state_matrix.float().unsqueeze(0).unsqueeze(0), (1,1,1,1), mode='circular')
        
        # 2. 计算每个位置的邻域合作者数量（4邻居+自身）
        neighbor_coop = F.conv2d(padded, self.neibor_kernel).squeeze()
        
        # 3. 计算每个群体 g 的红利：r * N_C^g / 5
        group_gross_profit = (self.r * neighbor_coop) / 5.0
        
        # 4. 对红利矩阵进行padding
        padded_gross = F.pad(group_gross_profit.unsqueeze(0).unsqueeze(0), (1,1,1,1), mode='circular')
        
        # 5. 计算每个智能体从参与的5个群体中获得的总红利
        total_gross_profit = F.conv2d(padded_gross, self.neibor_kernel).squeeze()
        
        # 6. 扣除成本（合作者扣5，背叛者扣0）
        total_cost = state_matrix.float() * 5.0
        
        # 7. 净收益
        reward_matrix = total_gross_profit - total_cost
        
        return reward_matrix

    def fermi_update(self, type_t_matrix):
        K = 0.1
        profit = self.calculate_reward(type_t_matrix)  # 现在profit已包含激励
        
        W_left = 1 / (1 + torch.exp((profit - torch.roll(profit, 1, 1))/K))
        W_right = 1 / (1 + torch.exp((profit - torch.roll(profit, -1, 1))/K))
        W_up = 1 / (1 + torch.exp((profit - torch.roll(profit, 1, 0))/K))
        W_down = 1 / (1 + torch.exp((profit - torch.roll(profit, -1, 0))/K))

        learning_direction = torch.randint(0,4,(self.L_num,self.L_num)).to(self.device)
        learning_probabilities = torch.rand(self.L_num,self.L_num).to(self.device)

        type_t1_matrix = (learning_direction==0)*((learning_probabilities<=W_left)*torch.roll(type_t_matrix,1,1)+(learning_probabilities>W_left)*type_t_matrix) +\
                        (learning_direction==1)*((learning_probabilities<=W_right)*torch.roll(type_t_matrix,-1,1)+(learning_probabilities>W_right)*type_t_matrix) +\
                        (learning_direction==2)*((learning_probabilities<=W_up)*torch.roll(type_t_matrix,1,0)+(learning_probabilities>W_up)*type_t_matrix) +\
                        (learning_direction==3)*((learning_probabilities<=W_down)*torch.roll(type_t_matrix,-1,0)+(learning_probabilities>W_down)*type_t_matrix)
        return type_t1_matrix.view(self.L_num,self.L_num)

    def run(self, num):
        coop_rates = []  # 合作率 = 当前状态中1的比例
        defect_rates = []  # 背叛率 = 当前状态中0的比例
        total_values = []  # 平均收益 = 所有个体的收益均值
        
        for epoch in tqdm(range(self.epoches)):
            self.epoch = epoch
            
            if epoch == 0:
                profit_matrix = self.calculate_reward(self.current_state)
                asyncio.create_task(self.shot_pic_with_ubp_style(self.current_state, epoch, self.r, profit_matrix))
                # 计算当前指标（新增部分）
                current_coop_rate = self.current_state.float().mean().item()  # 合作率
                current_defect_rate = 1 - current_coop_rate  # 背叛率
                current_profit = self.calculate_reward(self.current_state).mean().item()  # 平均收益
                
                # 记录指标（新增部分）
                coop_rates.append(current_coop_rate)
                defect_rates.append(current_defect_rate)
                total_values.append(current_profit)
            
            # Fermi 更新   
            self.current_state = self.fermi_update(self.current_state)  

            if (epoch+1 in [1, 10, 100, 1000, 10000, 100000]):
                profit_matrix = self.calculate_reward(self.current_state)
                asyncio.create_task(self.shot_pic_with_ubp_style(self.current_state, epoch+1, self.r, profit_matrix))

            # 计算当前指标（新增部分）
            current_coop_rate = self.current_state.float().mean().item()  # 合作率
            current_defect_rate = 1 - current_coop_rate  # 背叛率
            current_profit = self.calculate_reward(self.current_state).mean().item()  # 平均收益
            
            # 记录指标（新增部分）
            coop_rates.append(current_coop_rate)
            defect_rates.append(current_defect_rate)
            total_values.append(current_profit)

        return defect_rates, coop_rates, total_values

    def _save_snapshot(self, epoch, run_num):
        plt.figure(figsize=(8,8))
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        cmap = plt.get_cmap('Set1', 2)
        # 设置 figure 的边框为黑色
        fig.patch.set_edgecolor('black')
        fig.patch.set_linewidth(2)  # 设置边框线宽
        plt.imshow(self.current_state.cpu().numpy(), cmap='gray_r')
        plt.title(f"Epoch {epoch}, Coop Rate: {self.current_state.float().mean().item():.2f}")
        plt.savefig(f"{self.output_path}/snapshot_run{run_num}_epoch{epoch}.pdf", format='pdf', dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close()

    def save_data(self, data_type, name, r, run_num, data):
        output_dir = f'{self.output_path}/{data_type}'
        os.makedirs(output_dir, exist_ok=True)
        np.savetxt(f'{output_dir}/{name}_run{run_num}.txt', data)

    async def shot_pic_with_ubp_style(self, type_t_matrix, epoch, r, profit_data):
        """使用UBP-PPO风格的配色方案保存策略矩阵快照"""
        plt.clf()
        plt.close("all")
        
        # 创建输出目录
        img_dir = f'{self.output_path}/shot_pic/r={r}/two_type'
        matrix_dir = f'{self.output_path}/shot_pic/r={r}/two_type/type_t_matrix'
        profit_dir = f'{self.output_path}/shot_pic/r={r}/two_type/profit_matrix'
        
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(matrix_dir, exist_ok=True)
        os.makedirs(profit_dir, exist_ok=True)

        # 计算统计信息
        coop_rate = type_t_matrix.float().mean().item()
        defect_rate = 1 - coop_rate
        
        # 提取profit数据
        if isinstance(profit_data, tuple):
            profit_matrix = profit_data[0] if isinstance(profit_data[0], torch.Tensor) else torch.tensor(profit_data[0])
        else:
            profit_matrix = profit_data if isinstance(profit_data, torch.Tensor) else torch.tensor(profit_data)
        
        avg_profit = profit_matrix.mean().item()
        std_profit = profit_matrix.std().item()
        
        fig_dpi = 300

        # =============================================
        # 1. 保存策略矩阵图（与 LMF_MCPPO_DBP.py 一致）
        # =============================================
        fig1 = plt.figure(figsize=(8, 8))
        ax1 = fig1.add_subplot(1, 1, 1)
        ax1.axis('off')
        fig1.patch.set_edgecolor('black')
        fig1.patch.set_linewidth(2)

        # 使用与 LMF_MCPPO_DBP.py 相同的黑白配色
        color_map = {
            0: [0, 0, 0],  # 黑色（背叛）
            1: [1, 1, 1]   # 白色（合作）
        }

        # 转换为RGB图像
        strategy_image = np.zeros((self.L_num, self.L_num, 3))
        for label, color in color_map.items():
            strategy_image[type_t_matrix.cpu().numpy() == label] = color

        # 绘图设置
        ax1.imshow(strategy_image, interpolation='none')
        ax1.axis('off')
        for spine in ax1.spines.values():
            spine.set_linewidth(3)

        # 保存图片
        fig1.savefig(f'{img_dir}/strategy_distribution_t={epoch}.pdf', format='pdf', dpi=300, bbox_inches='tight', pad_inches=0)
        fig1.savefig(f'{img_dir}/strategy_distribution_t={epoch}.jpg', format='jpg', dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close(fig1)
        
        # =============================================
        # 2. 保存收益热图（与 LNMF_PPO 一致）
        # =============================================
        plt.figure(figsize=(8, 8))
        
        # 确保是numpy数组
        if isinstance(profit_matrix, torch.Tensor):
            profit_matrix = profit_matrix.cpu().numpy()
        
        # 与 LNMF_PPO 保持一致：颜色范围 0-9，但只显示 0-8 的刻度
        vmin = 0
        vmax = 9
        
        im = plt.imshow(profit_matrix, vmin=vmin, vmax=vmax, cmap='viridis', interpolation='none', aspect='equal')
        plt.axis('off')
        
        # 添加颜色条，只显示 0-8 的刻度
        cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=28)
        cbar.set_ticks(np.arange(0, 9, 1))  # 只显示0-8的刻度
        
        plt.tight_layout()
        
        # 同时保存 PDF 和 PNG
        pdf_path = f'{img_dir}/profit_t={epoch}.pdf'
        plt.savefig(pdf_path, format='pdf', dpi=fig_dpi, bbox_inches='tight', pad_inches=0)
        plt.close()

        # =============================================
        # 3. 保存矩阵数据文件
        # =============================================
        np.savetxt(f'{matrix_dir}/T{epoch}.txt',
                    type_t_matrix.cpu().numpy(), fmt='%d')
        np.savetxt(f'{profit_dir}/T{epoch}.txt',
                    profit_matrix, fmt='%.4f')
        return 0

    async def shot_pic(self, type_t_matrix, epoch, r, profit_data):
        """保留原有方法（向后兼容）"""
        return await self.shot_pic_with_ubp_style(type_t_matrix, epoch, r, profit_data)