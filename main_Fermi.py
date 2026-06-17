import torch
from Fermi import SPGG
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
def save_params_to_json(params, filename_prefix="params",output_path='data'):
    # 创建参数保存目录
    param_dir = Path(output_path)
    os.makedirs(output_path, exist_ok=True)
    # param_dir.mkdir(exist_ok=True)
    
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
    
    src_file='main_Fermi.py'
    dst_file=f'{output_path}/{src_file}'
    shutil.copy2(src_file, dst_file)
    src_file='Fermi.py'
    dst_file=f'{output_path}/{src_file}'
    shutil.copy2(src_file, dst_file)
    
    print(f"参数已保存至: {filepath}")

# 主实验程序
async def main(args):
    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y_%m_%d_%H%M%S")
    # 实验参数设置
    fontsize=16
    r_values = [round(i * 0.1, 1) for i in range(25, 56)]
    # 使用 arange 生成从 1 到 6 的列表，间隔为 0.1
    # r_values = [round(i * 0.1, 1) for i in range(45, 56)]
    # r_values = [round(i * 0.1, 1) for i in range(30, 61)]
    # print(result_list)

    if args.device=='cuda':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device=='cpu':
        device = torch.device("cpu")
    if args.epochs==1000:
        xticks=[0, 1, 10, 100, 1000]
    elif args.epochs==10000:
        xticks=[0, 1, 10, 100, 1000, 10000]
    elif args.epochs==100000:
        xticks=[0, 1, 10, 100, 1000, 10000, 100000]
    fra_yticks=[0.00, 0.20, 0.40, 0.60, 0.80, 1.00]
    profite_yticks=[0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

    # 实验参数设置
    experiment_params = {
        "r": r_values,
        "epochs": args.epochs,
        "runs": args.runs,
        "L_num": args.L_num,
        "xticks": xticks,
        "fra_yticks": fra_yticks,
        "profite_yticks": profite_yticks,
        "start_time": formatted_time
    }

    output_path=f'data/Fermi_{formatted_time}_q{str(args.question)}_e_{args.epochs}_L_{args.L_num}_seed_{args.seed}'
    save_params_to_json(experiment_params, filename_prefix="params",output_path=output_path)

    # 实验循环
    for r in r_values:
        print(f"\nRunning experiment with r={r}")
        # 多次独立运行
        for num in range(args.runs):
            # 初始化模型
            model = SPGG(
                L_num=args.L_num,
                device=device,
                r=r,
                epoches=args.epochs,
                now_time=formatted_time,
                question=args.question,
                output_path=output_path
            )
            print(f"Run {num+1}/{args.runs}")
            model.count = num  # 记录运行次数
            
            # 执行模拟
            D_Y, C_Y, all_value = model.run(num=num)
            
            # 保存实验结果
            model.save_data('Density_D', f'Density_D_r{r}', r, num, D_Y) # Density_D（背叛者密度），保存每个时间步中选择背叛策略的个体比例
            model.save_data('Density_C', f'Density_C_r{r}', r, num, C_Y) # Density_C（合作者密度），保存每个时间步中选择合作策略的个体比例
            model.save_data('Total_Value', f'Total_Value_r{r}', r, num, all_value) # Total_Value（系统总收益）,保存每个时间步中整个网格的总收益
            
                       # 绘制策略演化图
            plt.clf()
            plt.close("all")
            fig_dpi = 300

            # 在 main_Fermi.py 中，替换 strategy_evolution 绘图部分：

            # 子图1: 策略比例演化
            plt.figure(figsize=(8,6))

            # ===== 修改：横轴标签改为科学计数法上标格式 =====
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

            plt.plot(C_Y, 'b-', linewidth=2, alpha=0.7, label='C')
            plt.plot(D_Y, 'r-', linewidth=2, alpha=0.7, label='D')
            plt.xlim(0, None)
            plt.xscale('symlog', linthresh=1, linscale=0.5, subs=np.arange(1, 10))
            plt.xticks(xticks, xtick_labels, fontsize=28)  # 字号从20调大到28
            plt.yticks(fra_yticks, [f'{y:.1f}' for y in fra_yticks], fontsize=28)  # 字号从20调大到28
            plt.grid(True, which='both', linestyle='--', alpha=0.5)
            plt.xlabel('t', fontsize=28, labelpad=10)  # 字号从22调大到28
            plt.ylabel('Fractions', fontsize=28, labelpad=10)  # 字号从22调大到28
            plt.legend(loc='upper left', fontsize=28, framealpha=0.9)  

            plt.tight_layout()
            # 同时保存 PDF 和 PNG
            pdf_path = f'{output_path}/strategy_evolution_r{r}_run{num}.pdf'
            jpg_path = f'{output_path}/strategy_evolution_r{r}_run{num}.jpg'
            plt.savefig(pdf_path, format='pdf', dpi=fig_dpi, bbox_inches='tight', pad_inches=0.2)
            plt.savefig(jpg_path, format='jpg', dpi=fig_dpi, bbox_inches='tight', pad_inches=0.2)
            plt.close()

            # 子图2: 总收益演化
            plt.figure(figsize=(8,6))
            total_values_normalized = [v / (args.L_num * args.L_num) for v in all_value]
            plt.plot(total_values_normalized, 'g-', linewidth=2.5, alpha=0.8, label='Average Profit')
            plt.xlim(0, None)
            plt.xscale('symlog', linthresh=1, linscale=0.5, subs=np.arange(1, 10))
            plt.xticks(xticks, [str(x) for x in xticks], fontsize=14)
            plt.ylabel('Average Profit', fontsize=16, labelpad=10)
            plt.grid(True, which='both', linestyle='--', alpha=0.6)
            plt.legend(loc='upper right', fontsize=14, framealpha=0.9)

            final_profit = total_values_normalized[-1] if len(total_values_normalized) > 0 else 0
            plt.text(0.02, 0.98, f'Final: {final_profit:.3f}', 
                    transform=plt.gca().transAxes, verticalalignment='top', fontsize=12,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            plt.tight_layout()
            # 同时保存 PDF 和 PNG
            pdf_path = f'{output_path}/profit_evolution_r{r}_run{num}.pdf'
            plt.savefig(pdf_path, format='pdf', dpi=fig_dpi, bbox_inches='tight', pad_inches=0.2)
            plt.close()

            folder_path = f'{output_path}/Density_C'
            # 2. 获取所有 .txt 文件
            txt_files = [f for f in os.listdir(folder_path) if f.endswith(".txt")]

            # 3. 提取文件名中的数字和最后一行的数据
            x_values = []  # 存储文件名中的数字
            y_values = []  # 存储最后一行的数据

            for file_name in txt_files:
                # 提取文件名中的数字
                match = re.search(r"r(\d+\.\d+)", file_name)  # 匹配 r 后面的浮点数
                if match:
                    x_value = float(match.group(1))  # 提取数字并转换为整数
                    x_values.append(x_value)

                    # 提取最后一行的数据
                    file_path = os.path.join(folder_path, file_name)
                    with open(file_path, "r") as file:
                        lines = file.readlines()
                        if lines:  # 确保文件不为空
                            last_line = lines[-1].strip()  # 提取最后一行并去掉换行符
                            try:
                                y_value = float(last_line)  # 将最后一行转换为浮点数
                                y_values.append(y_value)
                            except ValueError:
                                print(f"文件 {file_name} 的最后一行不是有效的数字: {last_line}")

            # 4. 按 x 轴值排序
            sorted_data = sorted(zip(x_values, y_values), key=lambda x: x[0])
            x_values = [x[0] for x in sorted_data]
            y_values = [x[1] for x in sorted_data]
            y_values = np.array(y_values)

            # 5. 绘制折线图
            if x_values:
                plt.clf()
                plt.close("all")
                plt.figure(figsize=(8, 6))
                
                # ===== 修改：字号调大 =====
                plt.plot(x_values, y_values, marker="*", markersize=10, markerfacecolor='none', linestyle="-", color="b", label='C')
                plt.plot(x_values, 1-y_values, marker="p", markersize=10, markerfacecolor='none', linestyle="-", color="r", label='D')
                
                plt.xticks(fontsize=24) 
                plt.yticks(fontsize=24)  
                plt.legend(loc='best', fontsize=24)  
                plt.xlabel("r", fontsize=24)  
                plt.ylabel("Fractions", fontsize=24)  
                plt.ylim(0, 1)
                plt.grid(True)
                plt.savefig(f'{output_path}/C_D_r.png', 
                            dpi=300, bbox_inches='tight', pad_inches=0)
                plt.savefig(f'{output_path}/C_D_r.pdf', format='pdf', 
                            dpi=300, bbox_inches='tight', pad_inches=0)
                plt.close()
            else:
                print("没有找到有效的数据。")

    print("All experiments completed!")

if __name__ == "__main__":
    # 创建ArgumentParser对象
    parser = argparse.ArgumentParser(description='Process some parameters.')

    # 添加args参数
    # parser.add_argument('-r', type=float, default=0.5, help='R parameter')
    parser.add_argument('-epochs', type=int, default=10000, help='Epochs')
    parser.add_argument('-runs', type=int, default=1, help='Runs')
    parser.add_argument('-L_num', type=int, default=200, help='question size')
    parser.add_argument('-question', type=int, default=2, help='question')
    parser.add_argument('-eta', type=float, default=1.0, help='eta parameter')
    parser.add_argument('-alpha', type=float, default=0.3, help='alpha')
    parser.add_argument('-lam', type=float, default=0.5, help='lambda parameter')
    parser.add_argument('-device', type=str, default='cuda', help='Device')
    parser.add_argument('-is_not_cc', action='store_false', dest='is_cc', help='is not cc')
    parser.add_argument('-seed', type=int, default=1, help='random seed')

    # 解析命令行参数
    args = parser.parse_args()

    # AC_P:  r4.8:28,31,37,38,44,61,62,63
    # seed=2 # r5.0: 3,6,8,9,10, 
    # 固定 numpy 的随机数种子
    np.random.seed(args.seed)
    # 固定 PyTorch 的随机数种子
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # args.seed=seed
    asyncio.run(main(args))