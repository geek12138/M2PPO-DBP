# main_LMF_MAPPO_DP.py

import torch
from M2PPO_DBP import SPGG
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import json
from pathlib import Path
import asyncio
import argparse
import os
import shutil
import re

def save_params_to_json(params, filename_prefix="params", output_path='data'):
    # 创建参数保存目录
    param_dir = Path(output_path)
    os.makedirs(output_path, exist_ok=True)
    
    # 生成带时间戳的文件名
    timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    filename = f"{filename_prefix}_{timestamp}.json"
    filepath = param_dir / filename
    
    # 转换特殊数据类型
    serializable_params = {
        k: str(v) if isinstance(v, torch.device) else v  # 处理device类型
        for k, v in params.items()
    }
    
    # 保存到JSON
    with open(filepath, 'w') as f:
        json.dump(serializable_params, f, indent=4)
    
    src_file = 'main_LMF_MAPPO_DBP.py'
    dst_file = f'{output_path}/{src_file}'
    shutil.copy2(src_file, dst_file)
    src_file = 'LMF_MAPPO_DBP.py'
    dst_file = f'{output_path}/{src_file}'
    shutil.copy2(src_file, dst_file)

    print(f"参数已保存至: {filepath}")

# 主实验程序
async def main(args):
    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y_%m_%d_%H%M%S")
    # 实验参数设置
    fontsize = 16
    # 使用 arange 生成从 3.0 到 6.0 的列表，间隔为 0.1
    #r_values = [round(i * 0.1, 1) for i in range(25, 56)]
    r_values = [3.9,4.6]
    # 惩罚参数扫描（可单独运行不同p值）
    # p_values = args.p_punish_list if hasattr(args, 'p_punish_list') else [args.p_punish]
    p_values = [args.p_punish]

    if args.device == 'cuda':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == 'cpu':
        device = torch.device("cpu")
    elif args.device == 'mps':
        device = torch.device("mps")

    if args.epochs == 1000:
        xticks = [0, 1, 10, 100, 1000]
    elif args.epochs == 10000:
        xticks = [0, 1, 10, 100, 1000, 10000]
    elif args.epochs == 100000:
        xticks = [0, 1, 10, 100, 1000, 10000, 100000]
    fra_yticks = [0.0, 0.2, 0.4,  0.6,  0.8,  1.0]
    profite_yticks = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

    # 实验参数设置
    experiment_params = {
        "r": r_values,
        "epochs": args.epochs,
        "runs": args.runs,
        "L_num": args.L_num,
        "alpha": args.alpha,
        "gamma": args.gamma,
        "clip_epsilon": args.clip_epsilon,
        "question": args.question,
        "ppo_epochs": args.ppo_epochs,
        "batch_size": args.batch_size,
        "gae_lambda": args.gae_lambda,
        "device": device,
        "xticks": xticks,
        "fra_yticks": fra_yticks,
        "profite_yticks": profite_yticks,
        "start_time": formatted_time,
        "seed": args.seed,
        "delta": args.delta,
        "rho": args.rho,
        "p_punish": args.p_punish
    }

    output_path = f'data/LMF_MAPPO_DBP_{formatted_time}_q{str(args.question)}_e_{args.epochs}_L_{args.L_num}_a_{args.alpha}_g_{args.gamma}_ce_{args.clip_epsilon}_gl_{args.gae_lambda}_p_{args.ppo_epochs}_b_{args.batch_size}_delta_{args.delta}_rho_{args.rho}_p_{args.p_punish}_seed_{args.seed}'

    save_params_to_json(experiment_params, filename_prefix="params", output_path=output_path)

    # 实验循环
    for p_punish in p_values:
        for r in r_values:
            print(f"\nRunning experiment with r={r}, p_punish={p_punish}")
            # 多次独立运行
            for num in range(args.runs):
                # 初始化模型
                model = SPGG(
                    L_num=args.L_num,
                    device=device,
                    alpha=args.alpha,
                    gamma=args.gamma,
                    clip_epsilon=args.clip_epsilon,
                    r=r,
                    epochs=args.epochs,
                    now_time=formatted_time,
                    question=args.question,
                    ppo_epochs=args.ppo_epochs,
                    batch_size=args.batch_size,
                    gae_lambda=args.gae_lambda,
                    output_path=output_path,
                    delta=args.delta,
                    rho=args.rho,
                    p_punish=p_punish
                )
                print(f"Run {num+1}/{args.runs}")
                model.count = num
                
                # 执行模拟
                D_Y, C_Y, D_Value, C_Value, all_value = model.run()
                
                # 保存实验结果
                model.save_data('Density_D', f'r{r}', r, D_Y)
                model.save_data('Density_C', f'r{r}', r, C_Y)
                model.save_data('Value_D', f'r{r}', r, D_Value)
                model.save_data('Value_C', f'r{r}', r, C_Value)
                model.save_data('Total_Value', f'r{r}', r, all_value)

                # 修复后的绘图代码
                plt.clf()
                plt.close("all")

                # 先创建图形
                plt.figure(figsize=(8, 6))

                # 然后在图形上设置属性
                plt.yscale('linear')
                plt.grid(True, which='both', axis='y', linestyle='--', linewidth=0.5)
                plt.axhline(y=0.5, color='gray', linestyle=':', linewidth=1)

                # 绘制双曲线
                plt.plot(C_Y, 'b-', linewidth=2, alpha=0.7, label='C')
                plt.plot(D_Y, 'r-', linewidth=2, alpha=0.7, label='D')

                plt.xlim(0, None)
                plt.xscale('symlog',
                        linthresh=1,
                        linscale=0.5,
                        subs=np.arange(1, 10))

                # ===== 修改：横轴标签改为科学计数法上标格式 =====
                # 定义转换函数
                def format_xtick(x):
                    if x == 100:
                        return r'$10^2$'
                    elif x == 1000:
                        return r'$10^3$'
                    elif x == 10000:
                        return r'$10^4$'
                    elif x == 100000:
                        return r'$10^5$'
                    else:
                        return str(int(x))

                xtick_labels = [format_xtick(x) for x in xticks]
                plt.xticks(xticks, xtick_labels, fontsize=28)  
                plt.yticks(fra_yticks, fontsize=28)  

                plt.grid(True, which='both', linestyle='--', alpha=0.5)
                plt.xlabel('t', fontsize=28, labelpad=10) 
                plt.ylabel('Fractions', fontsize=28, labelpad=10) 

                plt.legend(loc='best', fontsize=28)  

                plt.savefig(f'{output_path}/strategy_evolution_r{r}_run{num}.pdf', format='pdf',
                            dpi=300, bbox_inches='tight', pad_inches=0)
                plt.savefig(f'{output_path}/strategy_evolution_r{r}_run{num}.jpg', format='jpg',
                            dpi=300, bbox_inches='tight', pad_inches=0)
                plt.close()

    print("All experiments completed!")

    # 绘制合作分数随r变化的曲线（跨多次运行的统计）
    folder_path = f'{output_path}/Density_C'
    txt_files = [f for f in os.listdir(folder_path) if f.endswith(".txt")]

    x_values = []
    y_values = []

    for file_name in txt_files:
        match = re.search(r"r(\d+\.\d+)", file_name)
        if match:
            x_value = float(match.group(1))
            x_values.append(x_value)

            file_path = os.path.join(folder_path, file_name)
            with open(file_path, "r") as file:
                lines = file.readlines()
                if lines:
                    last_line = lines[-1].strip()
                    try:
                        y_value = float(last_line)
                        y_values.append(y_value)
                    except ValueError:
                        print(f"文件 {file_name} 的最后一行不是有效的数字: {last_line}")

    # 按 x 轴值排序
    if x_values:
        # 转换为 numpy 数组以便进行向量化运算
        x_values_array = np.array(x_values)
        y_values_array = np.array(y_values)
        
        # 按 x 排序
        sort_idx = np.argsort(x_values_array)
        x_sorted = x_values_array[sort_idx]
        y_sorted = y_values_array[sort_idx]
        
        plt.clf()
        plt.close("all")
        plt.figure(figsize=(8, 6))
        
        plt.plot(x_sorted, y_sorted, marker="*", markersize=10, markerfacecolor='none', 
                linestyle="-", color="b", label='C')
        plt.plot(x_sorted, 1 - y_sorted, marker="p", markersize=10, markerfacecolor='none', 
                linestyle="-", color="r", label='D')
        
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)
        plt.legend(loc='best', fontsize=24)
        plt.xlabel("r", fontsize=24)
        plt.ylabel("Fractions", fontsize=24)
        plt.ylim(0, 1)
        plt.grid(True)
        
        plt.savefig(f'{output_path}/C_D_r.pdf', format='pdf', 
                    dpi=300, bbox_inches='tight', pad_inches=0)
        plt.savefig(f'{output_path}/C_D_r.jpg', format='jpg', 
                    dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close()
    else:
        print("没有找到有效的数据。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LMF-MAPPO-DP: Local Mean-Field MAPPO with Density-based Punishment')
    
    # 基本参数
    parser.add_argument('-epochs', type=int, default=1000, help='训练迭代次数')
    parser.add_argument('-runs', type=int, default=1, help='每个r值的独立运行次数')
    parser.add_argument('-L_num', type=int, default=200, help='网格大小 L x L')
    parser.add_argument('-alpha', type=float, default=1e-3, help='学习率')
    parser.add_argument('-gamma', type=float, default=0.99, help='折扣因子')
    parser.add_argument('-clip_epsilon', type=float, default=0.2, help='PPO裁剪阈值')
    parser.add_argument('-question', type=int, default=3, help='初始化方式: 1=随机, 2=半半, 3=全背叛, 4=棋盘格')
    parser.add_argument('-ppo_epochs', type=int, default=1, help='PPO更新轮数')
    parser.add_argument('-batch_size', type=int, default=1, help='批大小')
    parser.add_argument('-gae_lambda', type=float, default=0.95, help='GAE参数')
    parser.add_argument('-device', type=str, default='cuda', help='设备: cuda/cpu/mps')
    parser.add_argument('-seed', type=int, default=1, help='随机种子')
    parser.add_argument('-output_path', type=str, default='data', help='输出路径')
    parser.add_argument('-delta', type=float, default=0.5, help='价值损失权重')
    parser.add_argument('-rho', type=float, default=0.02, help='熵正则权重')
    
    # ===== LMF-MAPPO-DP 新增：密度惩罚参数 =====
    parser.add_argument('-p_punish', type=float, default=0.5, help='密度惩罚强度 p')
    
    args = parser.parse_args()
    
    # 固定随机种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    asyncio.run(main(args))