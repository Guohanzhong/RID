import json
import matplotlib.pyplot as plt
import os
# 读取 JSON 文件
def read_json(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

# 文件路径
files = ['./trained_lora_clean/avg_grad_dict.json', './zero_lora_clean/avg_grad_dict.json', 
        './trained_lora_clean/avg_loss_dict.json', './zero_lora_clean/avg_loss_dict.json']

files = ['./trained_lora_gam/avg_grad_dict.json', './zero_lora_gam/avg_grad_dict.json', 
        './trained_lora_gam/avg_loss_dict.json', './zero_lora_gam/avg_loss_dict.json','./trained_lora_clean/avg_grad_dict.json', './zero_lora_clean/avg_grad_dict.json', 
        './trained_lora_clean/avg_loss_dict.json', './zero_lora_clean/avg_loss_dict.json']

def extract_key(file_path):
    folder_name = os.path.split(os.path.dirname(file_path))[1]
    file_name = os.path.basename(file_path)
    if 'grad' in file_name:
        key = f"{folder_name}_grad"
    elif 'loss' in file_name:
        key = f"{folder_name}_loss"
    else:
        key = folder_name
    return key
# 读取所有文件
keys = [extract_key(file) for file in files]
data = {key: read_json(file) for key, file in zip(keys, files)}

# 提取时间步和值
time_steps = list(range(1, 1000))
values = {key: [data[key][str(step)] for step in time_steps] for key in keys}
print(values.keys())

# 绘制并保存图表
def plot_and_save_values(values1, values2, label1, label2, save_path,values3=None,values4=None,label3=None,label4=None):
    plt.figure(figsize=(12, 6))
    plt.plot(time_steps, values1, label=label1)
    plt.plot(time_steps, values2, label=label2)
    if values3 is not None:
        plt.plot(time_steps, values3, label=label3)
    if values4 is not None:
        plt.plot(time_steps, values4, label=label4)
    plt.xlabel('Time Steps')
    plt.ylabel('Values')
    #plt.title(f'{label1} vs {label2}')
    plt.legend()
    plt.savefig(save_path)
    plt.close()

plot_and_save_values(values['trained_lora_gam_grad'], values['zero_lora_gam_grad'], 'gam_trained', 'gam_initiated', 'avg_grad_comparison2.png',
                    values3=values['trained_lora_clean_grad'],values4=values['zero_lora_clean_grad'],label3='clean_trained',label4='clean_initiated')
plot_and_save_values(values['trained_lora_gam_loss'], values['zero_lora_gam_loss'], 'gam_trained', 'gam_initiated', 'avg_loss_comparison2.png',
                    values3=values['trained_lora_clean_loss'],values4=values['zero_lora_clean_loss'],label3='clean_trained',label4='clean_initiated')

# 生成并保存两个图表
#plot_and_save_values(values['trained_lora_clean_grad'], values['zero_lora_clean_grad'], 'clean_trained', 'clean_initiated', 'avg_grad_comparison2.png')
#plot_and_save_values(values['trained_lora_clean_loss'], values['zero_lora_clean_loss'], 'avg_loss_dict', 'avg_loss_dict_c', 'avg_loss_comparison2.png')
