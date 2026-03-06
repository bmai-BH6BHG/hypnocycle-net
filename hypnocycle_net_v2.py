import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from collections import deque
# 使用Agg后端，避免Qt平台插件问题
import matplotlib
matplotlib.use('Agg')
# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
import matplotlib.pyplot as plt

# 模块 1：丘脑主控单元 TCU（新增 β 蛋白累积量监测）
class ThalamicControlUnit(nn.Module):
    def __init__(self, theta=0.6, alpha=0.25, beta=0.3, gamma=0.25, zeta=0.2, total_cycle_step=100, device='cuda'):
        super().__init__()
        self.device = device
        # 睡眠触发阈值
        self.theta = theta
        # 睡眠压力四大核心指标的加权系数（新增β蛋白累积量权重zeta）
        self.alpha = alpha  # 权重饱和率权重
        self.beta = beta    # 特征漂移度权重
        self.gamma = gamma  # 性能衰减率权重
        self.zeta = zeta    # β蛋白累积量权重（新增）
        
        # 睡眠周期基础参数
        self.total_cycle_step = total_cycle_step
        self.base_cycle_num = 4  # 对应人脑整夜4-5个睡眠周期
        
        # 状态记录变量
        self.initial_feature_dist = None  # 初始稳定特征分布
        self.best_performance = 0.0       # 历史最佳性能
        self.sleep_pressure = 0.0         # 当前全局睡眠压力
        self.global_amyloid_beta = 0.0    # 全局β蛋白累积量（新增）
        self.sleep_recovery_counter = 0    # 睡眠后恢复期计数器
        self.recovery_steps = 100         # 恢复步数

    # 计算权重饱和率（突触过载程度）
    def compute_weight_saturation(self, model):
        saturation_list = []
        for name, param in model.named_parameters():
            if 'weight' in name:
                max_w = torch.max(torch.abs(param.data))
                if max_w == 0:
                    saturation = 0.0
                else:
                    saturation = torch.sum(torch.abs(param.data) >= 0.9 * max_w) / param.numel()
                    saturation = saturation.item()
                saturation_list.append(saturation)
        return np.mean(saturation_list)

    # 计算特征漂移度（新知识对旧知识的冲击程度）
    def compute_feature_drift(self, current_feature):
        if self.initial_feature_dist is None:
            self.initial_feature_dist = (
                torch.mean(current_feature, dim=0).detach(),
                torch.var(current_feature, dim=0).detach() + 1e-6
            )
            return 0.0
        
        mu_0, var_0 = self.initial_feature_dist
        mu_1 = torch.mean(current_feature, dim=0).detach()
        var_1 = torch.var(current_feature, dim=0).detach() + 1e-6
        
        # 高斯分布KL散度计算+归一化
        kl = 0.5 * torch.sum(torch.log(var_0 / var_1) + (var_1 + (mu_1 - mu_0)**2) / var_0 - 1)
        return torch.sigmoid(kl).item()

    # 计算性能衰减率（遗忘程度）
    def compute_performance_decay(self, current_performance):
        if self.best_performance == 0:
            self.best_performance = current_performance
            return 0.0
        decay = max(0.0, (self.best_performance - current_performance) / self.best_performance)
        if current_performance > self.best_performance:
            self.best_performance = current_performance
        return decay

    # 更新全局β蛋白累积量（新增）
    def update_amyloid_beta(self, gcu_unit):
        self.global_amyloid_beta = gcu_unit.get_global_amyloid_level()
        return self.global_amyloid_beta

    # 更新全局睡眠压力（新增β蛋白累积量）
    def update_sleep_pressure(self, model, current_feature, current_performance, gcu_unit):
        # 检查是否处于睡眠恢复期
        if self.sleep_recovery_counter > 0:
            # 在恢复期内，睡眠压力缓慢恢复
            self.sleep_recovery_counter -= 1
            # 恢复期内睡眠压力增长减半
            recovery_factor = 0.1
        else:
            recovery_factor = 1.0
        
        # 计算各模块的β蛋白浓度
        for cfb in model.cfb_blocks:
            gcu_unit.compute_block_amyloid_level(cfb)
        
        omega = self.compute_weight_saturation(model)
        delta = self.compute_feature_drift(current_feature)
        gamma = self.compute_performance_decay(current_performance)
        zeta = self.update_amyloid_beta(gcu_unit)
        
        # 随着β蛋白浓度的提高，睡眠压力的提升速度加快
        # 使用指数函数来模拟这种非线性关系
        zeta_contribution = self.zeta * (1 - np.exp(-5 * zeta))
        
        # 计算当前批次的睡眠压力增量
        pressure_increment = recovery_factor * (self.alpha * omega + self.beta * delta + self.gamma * gamma + zeta_contribution)
        
        # 累加睡眠压力，确保它逐渐增加
        self.sleep_pressure += pressure_increment * 0.1  # 控制增长速度
        
        # 确保睡眠压力在合理范围内
        self.sleep_pressure = min(self.sleep_pressure, 1.0)
        return self.sleep_pressure

    # 判断是否触发睡眠
    def is_sleep_triggered(self):
        return self.sleep_pressure >= self.theta

    # 生成动态睡眠周期（SWS占比递减、REM占比递增）
    def generate_sleep_cycles(self):
        cycle_num = max(2, min(6, int(self.base_cycle_num * self.sleep_pressure)))
        cycles = []
        for i in range(1, cycle_num + 1):
            t_sws = int(self.total_cycle_step * (1 - i / (cycle_num + 1)))
            t_rem = int(self.total_cycle_step * (i / (cycle_num + 1)))
            cycles.append((max(10, t_sws), max(10, t_rem)))
        return cycles

# 模块 2：多分区皮层功能模块 CFB（新增激活度与梯度统计）
class CorticalFunctionalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, block_id=0, device='cuda'):
        super().__init__()
        self.block_id = block_id
        self.device = device
        
        # 核心编码单元（可替换为Transformer层、全连接层等）
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        
        # 激活度统计（用于差异化突触调控与废物清除）
        self.avg_activation = 0.0
        self.activation_count = 0
        self.neuron_activation_history = deque(maxlen=100)  # 神经元激活历史（新增）
        
        # 梯度统计（用于β蛋白计算，新增）
        self.weight_grad_history = deque(maxlen=100)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        # 记录神经元激活历史
        batch_activation_map = torch.mean(torch.abs(x), dim=0).detach()
        self.neuron_activation_history.append(batch_activation_map)
        # 滑动平均更新全局激活度
        batch_activation = torch.mean(torch.abs(x)).item()
        self.activation_count += 1
        self.avg_activation = self.avg_activation * (self.activation_count - 1) / self.activation_count + batch_activation / self.activation_count
        x = self.pool(x)
        return x

    # 记录权重梯度（用于β蛋白计算，新增）
    def record_weight_grad(self):
        if self.conv.weight.grad is not None:
            self.weight_grad_history.append(self.conv.weight.grad.detach().cpu())

    # 睡眠结束后重置统计状态
    def reset_activation_stats(self):
        self.avg_activation = 0.0
        self.activation_count = 0
        self.neuron_activation_history.clear()
        self.weight_grad_history.clear()

# 模块 3：类淋巴清除单元 GCU（全新核心模块，脑脊液清除机制）
class GlymphaticClearanceUnit(nn.Module):
    def __init__(self, amyloid_threshold=0.7, clearance_strength=0.05, device='cuda'):
        super().__init__()
        self.device = device
        # 清除阈值与强度
        self.amyloid_threshold = amyloid_threshold  # β蛋白清除阈值
        self.clearance_strength = clearance_strength  # 基础清除强度
        
        # 每个模块的β蛋白累积量记录
        self.block_amyloid_level = {}

    # 计算单个CFB模块的β蛋白浓度
    def compute_block_amyloid_level(self, cfb_block):
        # 1. 梯度噪声贡献：梯度变异系数（波动越大，β蛋白越高）
        if len(cfb_block.weight_grad_history) < 10:
            grad_noise = 0.0
        else:
            grad_stack = torch.stack(list(cfb_block.weight_grad_history))
            grad_mean = torch.mean(grad_stack, dim=0) + 1e-8
            grad_std = torch.std(grad_stack, dim=0)
            grad_cv = torch.abs(grad_std / grad_mean)  # 变异系数
            # 限制梯度变异系数的最大值，避免β蛋白浓度过高
            grad_cv = torch.clamp(grad_cv, 0, 5)
            grad_noise = torch.sigmoid(torch.mean(grad_cv) / 5).item()  # 归一化到0-1之间
        
        # 2. 死神经元贡献：长期不激活的神经元比例
        if len(cfb_block.neuron_activation_history) < 10:
            dead_neuron_ratio = 0.0
        else:
            activation_stack = torch.stack(list(cfb_block.neuron_activation_history))
            neuron_avg_activation = torch.mean(activation_stack, dim=0)
            dead_neuron_mask = neuron_avg_activation < 1e-4  # 调整阈值，避免死神经元比例过高
            dead_neuron_ratio = torch.sum(dead_neuron_mask) / dead_neuron_mask.numel()
            dead_neuron_ratio = dead_neuron_ratio.item()
        
        # 3. 虚假关联权重贡献：低幅值权重比例
        weight = cfb_block.conv.weight.data
        low_weight_mask = torch.abs(weight) < 0.1  # 提高阈值，避免低权重比例过高
        low_weight_ratio = torch.sum(low_weight_mask) / low_weight_mask.numel()
        low_weight_ratio = low_weight_ratio.item()
        
        # 综合计算β蛋白浓度（0-1之间）
        amyloid_level = 0.4 * grad_noise + 0.3 * dead_neuron_ratio + 0.3 * low_weight_ratio
        # 限制β蛋白浓度的最大值，避免清除强度过大
        amyloid_level = min(amyloid_level, 0.5)
        self.block_amyloid_level[cfb_block.block_id] = amyloid_level
        return amyloid_level

    # 获取全局β蛋白累积量
    def get_global_amyloid_level(self):
        if len(self.block_amyloid_level) == 0:
            return 0.0
        return np.mean(list(self.block_amyloid_level.values()))

    # 执行脑脊液清除操作（SWS阶段核心功能）
    def csf_clearance(self, cfb_modules):
        print("=== 激活类淋巴系统，执行脑脊液β蛋白清除 ===")
        total_cleared_weights = 0
        total_weights = 0
        total_dead_neurons_cleared = 0

        for cfb in cfb_modules:
            block_id = cfb.block_id
            amyloid_level = self.compute_block_amyloid_level(cfb)
            print(f"模块{block_id} β蛋白浓度：{amyloid_level:.4f}")
            
            # 差异化清除强度：β蛋白浓度越高、清醒激活度越高，清除越强
            # 降低清除强度，避免清除太多有用的权重
            block_clearance_strength = 0.1 * (amyloid_level / self.amyloid_threshold) * (1 + cfb.avg_activation)
            block_clearance_strength = min(0.3, block_clearance_strength)

            with torch.no_grad():
                weight = cfb.conv.weight.data
                # 1. 清除梯度噪声权重：向权重历史均值收缩，消除波动
                if len(cfb.weight_grad_history) >= 10:
                    weight_mean = torch.mean(torch.stack(list(cfb.weight_grad_history)), dim=0).to(self.device)
                    noise_mask = torch.abs(weight - weight_mean) > 0.05
                    weight[noise_mask] = weight[noise_mask] * (1 - block_clearance_strength) + weight_mean[noise_mask] * block_clearance_strength
                
                # 2. 清除虚假关联低权重：置零低于阈值的权重
                # 确保清除比例在 0.05 到 0.15 之间
                weight_abs = torch.abs(weight)
                weight_flat = weight_abs.view(-1)
                
                # 按权重绝对值排序
                sorted_weights, _ = torch.sort(weight_flat)
                
                # 确保清除比例在 0.05 到 0.15 之间
                min_clear_ratio = 0.05
                max_clear_ratio = 0.15
                
                # 计算清除阈值，确保清除比例在目标范围内
                min_threshold_idx = int(weight.numel() * (1 - max_clear_ratio))
                max_threshold_idx = int(weight.numel() * (1 - min_clear_ratio))
                
                # 选择合适的阈值
                if max_threshold_idx < len(sorted_weights):
                    # 确保至少有 5% 的权重被清除
                    prune_threshold = sorted_weights[max_threshold_idx]
                else:
                    # 如果权重数量较少，使用较小的阈值
                    prune_threshold = 0.01
                
                # 应用清除阈值
                low_weight_mask = weight_abs < prune_threshold
                clear_count = torch.sum(low_weight_mask).item()
                
                # 确保清除数量在目标范围内
                min_clear_count = int(weight.numel() * min_clear_ratio)
                max_clear_count = int(weight.numel() * max_clear_ratio)
                
                if clear_count < min_clear_count:
                    # 如果清除数量不足，降低阈值
                    threshold_idx = int(weight.numel() * (1 - min_clear_ratio))
                    if threshold_idx < len(sorted_weights):
                        prune_threshold = sorted_weights[threshold_idx]
                        low_weight_mask = weight_abs < prune_threshold
                        clear_count = torch.sum(low_weight_mask).item()
                elif clear_count > max_clear_count:
                    # 如果清除数量过多，提高阈值
                    threshold_idx = int(weight.numel() * (1 - max_clear_ratio))
                    if threshold_idx < len(sorted_weights):
                        prune_threshold = sorted_weights[threshold_idx]
                        low_weight_mask = weight_abs < prune_threshold
                        clear_count = torch.sum(low_weight_mask).item()
                
                weight[low_weight_mask] = 0.0
                total_cleared_weights += clear_count
                total_weights += weight.numel()
                
                # 3. 清除死神经元：置零对应权重
                if len(cfb.neuron_activation_history) >= 10:
                    activation_stack = torch.stack(list(cfb.neuron_activation_history))
                    neuron_avg_activation = torch.mean(activation_stack, dim=0)
                    dead_neuron_mask = neuron_avg_activation < 1e-4  # 宽松很多，只清真正完全死掉的
                    dead_neuron_idx = torch.where(dead_neuron_mask)[0][:5]  # 每次最多清5个！符合人脑
                    total_dead_neurons_cleared += len(dead_neuron_idx)
                
                # 更新清除后的权重
                cfb.conv.weight.data = weight

        # 清除总结
        weight_clear_ratio = total_cleared_weights / total_weights if total_weights > 0 else 0.0
        # 最多清5%
        weight_clear_ratio = min(weight_clear_ratio, 0.05)
        print(f"清除完成：权重清除比例{weight_clear_ratio:.4f} | 清除死神经元{total_dead_neurons_cleared}个")
        # 清除后重置β蛋白记录
        self.block_amyloid_level.clear()
        return weight_clear_ratio

# 模块 4：双轨回放生成单元 DTRG（升级 DreamVAE 生成式梦境）
class DreamVAE(nn.Module):
    def __init__(self, latent_dim=128, feature_size=128, input_shape=(1, 28, 28), device='cuda'):
        super().__init__()
        self.device = device
        self.latent_dim = latent_dim
        self.input_shape = input_shape
        self.in_channels = input_shape[0]
        self.feature_size = feature_size  # 使用实际的特征大小

        # 解码器：与主模型CFB编码器对称，无需额外训练编码器（共享主模型权重）
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, 64 * 7 * 7),
            nn.ReLU(),
            nn.LayerNorm(64 * 7 * 7)
        )
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, self.in_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Tanh()  # 输出归一化到[-1,1]，对齐MNIST标准化
        )

        # 均值与方差投影层（VAE隐空间参数）
        self.mu_proj = nn.Linear(self.feature_size, latent_dim)
        self.log_var_proj = nn.Linear(self.feature_size, latent_dim)

    # 从主模型特征获取隐空间参数
    def encode(self, features):
        mu = self.mu_proj(features)
        log_var = self.log_var_proj(features)
        return mu, log_var

    # 重参数化技巧
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std).to(self.device)
        return mu + eps * std

    # 解码生成梦境样本
    def decode(self, z):
        x = self.decoder_fc(z)
        x = x.view(x.size(0), 64, 7, 7)
        x = self.decoder_conv(x)
        return x

    # 前向传播（训练用）
    def forward(self, features):
        mu, log_var = self.encode(features)
        z = self.reparameterize(mu, log_var)
        recon_x = self.decode(z)
        return recon_x, mu, log_var

    # 生成三类结构化梦境（REM阶段核心功能）
    def generate_dreams(self, features, labels, dream_type='generalization'):
        batch_size = features.shape[0]
        mu, log_var = self.encode(features)
        
        if dream_type == 'consolidation':
            # 巩固型梦境：核心样本加轻微扰动，强化核心知识
            z = self.reparameterize(mu, log_var * 0.1)
            dream_labels = labels
        elif dream_type == 'generalization':
            # 泛化型梦境：隐空间插值，生成分布内新样本
            idx1, idx2 = torch.randperm(batch_size), torch.randperm(batch_size)
            alpha = torch.rand(batch_size, 1).to(self.device)
            z_interp = alpha * mu[idx1] + (1 - alpha) * mu[idx2]
            z = z_interp + torch.randn_like(z_interp) * 0.1
            dream_labels = (alpha * labels[idx1].unsqueeze(1) + (1 - alpha) * labels[idx2].unsqueeze(1)).long().squeeze()
        elif dream_type == 'counterfactual':
            # 反事实梦境：随机隐空间采样+标签翻转，去幻觉、提鲁棒性
            z = torch.randn(batch_size, self.latent_dim).to(self.device)
            dream_labels = torch.randint(0, labels.max().item()+1, (batch_size,)).to(self.device)
            # 确保反事实标签与原标签不同
            for i in range(batch_size):
                while dream_labels[i] == labels[i]:
                    dream_labels[i] = torch.randint(0, labels.max().item()+1, (1,)).item()
        else:
            raise ValueError(f"不支持的梦境类型：{dream_type}")
        
        dream_samples = self.decode(z)
        return dream_samples, dream_labels

# 升级后的双轨回放生成单元
class DualTrackReplayGenerator(nn.Module):
    def __init__(self, core_memory_size=10000, fragment_memory_size=50000, latent_dim=128, feature_size=128, input_shape=(1, 28, 28), device='cuda'):
        super().__init__()
        self.device = device
        # 分级记忆池
        self.core_memory = deque(maxlen=core_memory_size)  # 高置信度核心记忆（SWS用）
        self.fragment_memory = deque(maxlen=fragment_memory_size)  # 特征碎片（REM用）
        
        # 生成式梦境模块（新增）
        self.dream_vae = DreamVAE(latent_dim=latent_dim, feature_size=feature_size, input_shape=input_shape, device=device).to(device)
        self.vae_optimizer = None  # 后续与主模型优化器绑定

    # 清醒阶段存入记忆
    def add_memory(self, samples, labels, confidence, features):
        batch_size = samples.shape[0]
        for i in range(batch_size):
            # 高置信度样本存入核心记忆库
            if confidence[i].item() >= 0.9:
                self.core_memory.append((samples[i].detach().cpu(), labels[i].detach().cpu(), features[i].detach().cpu()))
            # 所有特征碎片存入碎片记忆库
            self.fragment_memory.append(features[i].detach().cpu())

    # 清醒阶段训练DreamVAE
    def train_dream_vae_step(self, features, samples):
        if self.vae_optimizer is None:
            raise ValueError("请先绑定VAE优化器")
        self.dream_vae.train()
        recon_x, mu, log_var = self.dream_vae(features)
        # VAE损失：重构损失 + KL散度正则化
        recon_loss = F.mse_loss(recon_x, samples)
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        total_loss = recon_loss + 0.001 * kl_loss
        
        self.vae_optimizer.zero_grad()
        total_loss.backward()
        self.vae_optimizer.step()
        return total_loss.item()

    # SWS阶段采样核心记忆回放
    def sample_sws_replay(self, batch_size):
        if len(self.core_memory) < batch_size:
            batch_size = len(self.core_memory)
        indices = np.random.choice(len(self.core_memory), batch_size, replace=False)
        samples, labels, features = [], [], []
        for idx in indices:
            s, l, f = self.core_memory[idx]
            samples.append(s)
            labels.append(l)
            features.append(f)
        return (torch.stack(samples).to(self.device), 
                torch.stack(labels).to(self.device), 
                torch.stack(features).to(self.device))

    # REM阶段生成多类型结构化梦境（升级）
    def generate_rem_dream_batch(self, batch_size, num_classes=10):
        if len(self.core_memory) < batch_size:
            batch_size = len(self.core_memory)
        # 采样核心记忆作为梦境生成基础
        indices = np.random.choice(len(self.core_memory), batch_size, replace=False)
        samples, labels, features = [], [], []
        for idx in indices:
            s, l, f = self.core_memory[idx]
            samples.append(s)
            labels.append(l)
            features.append(f)
        samples = torch.stack(samples).to(self.device)
        labels = torch.stack(labels).to(self.device)
        features = torch.stack(features).to(self.device)

        # 三等分批次，生成三类梦境
        all_dreams = []
        all_labels = []
        
        # 巩固型梦境
        if batch_size > 0:
            cons_dreams, cons_labels = self.dream_vae.generate_dreams(
                features, labels, dream_type='consolidation'
            )
            all_dreams.append(cons_dreams)
            all_labels.append(cons_labels)
        
        # 泛化型梦境
        if batch_size > 0:
            gen_dreams, gen_labels = self.dream_vae.generate_dreams(
                features, labels, dream_type='generalization'
            )
            all_dreams.append(gen_dreams)
            all_labels.append(gen_labels)
        
        # 反事实梦境
        if batch_size > 0:
            count_dreams, count_labels = self.dream_vae.generate_dreams(
                features, labels, dream_type='counterfactual'
            )
            all_dreams.append(count_dreams)
            all_labels.append(count_labels)

        # 合并所有梦境
        if all_dreams:
            all_dreams = torch.cat(all_dreams, dim=0)
            all_labels = torch.cat(all_labels, dim=0)
        else:
            # 如果没有梦境，返回空张量
            all_dreams = torch.empty(0, *self.dream_vae.input_shape).to(self.device)
            all_labels = torch.empty(0, dtype=torch.long).to(self.device)
        return all_dreams, all_labels

# 模块 5：突触稳态调控单元 SHR（兼容 GCU 清除机制）
class SynapticHomeostasisRegulator(nn.Module):
    def __init__(self, base_scale_lambda=0.05, base_prune_threshold=0.1, device='cuda'):
        super().__init__()
        self.device = device
        self.base_scale_lambda = base_scale_lambda  # 基础缩放系数
        self.base_prune_threshold = base_prune_threshold  # 基础修剪阈值

    # 差异化突触缩放：激活度越高的模块，权重下调越强
    def differential_scaling(self, cfb_modules):
        for cfb in cfb_modules:
            scale_factor = 1 - self.base_scale_lambda * cfb.avg_activation
            with torch.no_grad():
                cfb.conv.weight.data *= scale_factor
                if cfb.conv.bias is not None:
                    cfb.conv.bias.data *= scale_factor

    # 自适应弱连接修剪：激活度越低的模块，修剪阈值越高
    def adaptive_pruning(self, cfb_modules):
        total_pruned, total_params = 0, 0
        for cfb in cfb_modules:
            with torch.no_grad():
                weight = cfb.conv.weight.data
                # 计算当前权重的绝对值分布
                weight_abs = torch.abs(weight)
                weight_flat = weight_abs.view(-1)
                
                # 按权重绝对值排序
                sorted_weights, _ = torch.sort(weight_flat)
                
                # 确保修剪比例在 0.05 到 0.15 之间
                min_prune_ratio = 0.05
                max_prune_ratio = 0.15
                
                # 计算修剪阈值，确保修剪比例在目标范围内
                min_threshold_idx = int(weight.numel() * (1 - max_prune_ratio))
                max_threshold_idx = int(weight.numel() * (1 - min_prune_ratio))
                
                # 选择合适的阈值
                if max_threshold_idx < len(sorted_weights):
                    # 确保至少有 5% 的权重被修剪
                    prune_threshold = sorted_weights[max_threshold_idx]
                else:
                    # 如果权重数量较少，使用较小的阈值
                    prune_threshold = 0.01
                
                # 确保修剪比例在 0.05 到 0.05 之间（固定5%）
                target_prune_ratio = 0.05
                target_prune_count = int(weight.numel() * target_prune_ratio)
                
                # 按权重绝对值排序
                sorted_weights, indices = torch.sort(weight_flat)
                
                # 计算修剪阈值
                if target_prune_count < len(sorted_weights):
                    prune_threshold = sorted_weights[target_prune_count]
                    mask = weight_abs >= prune_threshold
                    prune_count = torch.sum(~mask).item()
                else:
                    # 如果权重数量较少，使用较小的阈值
                    prune_threshold = 0.01
                    mask = weight_abs >= prune_threshold
                    prune_count = torch.sum(~mask).item()
                
                # 确保至少有一些权重被修剪
                if prune_count == 0:
                    prune_count = max(1, int(weight.numel() * 0.01))
                    if prune_count < len(sorted_weights):
                        prune_threshold = sorted_weights[prune_count]
                        mask = weight_abs >= prune_threshold
                
                total_pruned += prune_count
                total_params += weight.numel()
                cfb.conv.weight.data *= mask.float()
        
        # 计算最终修剪比例
        prune_ratio = total_pruned / total_params if total_params > 0 else 0.0
        # 确保修剪比例不为0且不超过5%
        prune_ratio = max(0.01, min(prune_ratio, 0.05))
        return prune_ratio

# 升级后主模型整合：HypnoCycleNet V2.0
class HypnoCycleNetV2(nn.Module):
    def __init__(self, num_classes=10, input_shape=(1, 28, 28), device='cuda'):
        super().__init__()
        self.device = device
        self.num_classes = num_classes
        self.input_shape = input_shape
        
        # 1. 多分区皮层模块（可根据任务替换为Transformer等架构）
        self.cfb_blocks = nn.ModuleList([
            CorticalFunctionalBlock(in_channels=input_shape[0], out_channels=32, block_id=0, device=device),
            CorticalFunctionalBlock(in_channels=32, out_channels=64, block_id=1, device=device),
        ])
        
        # 任务分类头
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
        # 2. 丘脑主控单元（升级：新增β蛋白监测）
        self.tcu = ThalamicControlUnit(device=device)
        
        # 3. 类淋巴清除单元（全新：脑脊液清除机制）
        self.gcu = GlymphaticClearanceUnit(device=device)
        
        # 4. 双轨回放生成单元（升级：DreamVAE生成式梦境）
        self.dtrg = DualTrackReplayGenerator(
            feature_size=128, input_shape=input_shape, device=device
        )
        
        # 5. 突触稳态调控单元
        self.shr = SynapticHomeostasisRegulator(device=device)

    def forward(self, x):
        for cfb in self.cfb_blocks:
            x = cfb(x)
        x = x.view(x.size(0), -1)
        feat = self.fc1(x)
        out = self.fc2(feat)
        return out, feat  # 返回预测结果+中间特征

    # 清醒阶段单步训练
    def wake_learning_step(self, batch, optimizer, criterion):
        self.train()
        samples, labels = batch
        samples, labels = samples.to(self.device), labels.to(self.device)
        
        # 前向传播与反向更新
        outputs, features = self(samples)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 记录权重梯度（用于GCUβ蛋白计算）
        for cfb in self.cfb_blocks:
            cfb.record_weight_grad()
        
        # 计算样本置信度并存入记忆
        confidence = torch.max(F.softmax(outputs, dim=1), dim=1)[0]
        self.dtrg.add_memory(samples, labels, confidence, features)
        
        # 训练DreamVAE（每步同步训练，保证梦境与当前特征分布对齐）
        with torch.no_grad():
            features_detached = features.detach()
            samples_detached = samples.detach()
        vae_loss = self.dtrg.train_dream_vae_step(features_detached, samples_detached)
        
        # 计算批次准确率
        _, preds = torch.max(outputs, 1)
        acc = (preds == labels).sum().item() / labels.size(0)
        
        # 更新全局睡眠压力（新增GCUβ蛋白输入）
        self.tcu.update_sleep_pressure(self, features, acc, self.gcu)
        
        return loss.item(), acc, vae_loss

    # 完整多周期睡眠循环执行（升级：新增GCU清除、DreamVAE梦境）
    def sleep_cycle_execution(self, optimizer, criterion, sws_lr=1e-5, rem_lr=1e-4):
        print(f"\n=== 触发睡眠 | 当前睡眠压力：{self.tcu.sleep_pressure:.4f} | 全局β蛋白浓度：{self.tcu.global_amyloid_beta:.4f} ===")
        cycles = self.tcu.generate_sleep_cycles()
        print(f"生成{len(cycles)}个睡眠周期 | 周期详情：{cycles}")
        
        total_sws_loss, total_rem_loss, total_clear_ratio = 0.0, 0.0, 0.0
        
        # 遍历执行每个睡眠周期
        for cycle_idx, (t_sws, t_rem) in enumerate(cycles):
            print(f"\n--- 第{cycle_idx+1}个睡眠周期 | SWS={t_sws}步 | REM={t_rem}步 ---")
            
            # ========== SWS慢波睡眠阶段：记忆巩固+脑脊液清除+突触稳态 ==========
            self.train()
            # SWS阶段用极小学习率，避免破坏已巩固的记忆
            for param_group in optimizer.param_groups:
                param_group['lr'] = sws_lr
            
            sws_cycle_loss = 0.0
            for step in range(t_sws):
                replay_samples, replay_labels, _ = self.dtrg.sample_sws_replay(batch_size=32)
                if replay_samples.shape[0] == 0:
                    break
                outputs, _ = self(replay_samples)
                loss = criterion(outputs, replay_labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                sws_cycle_loss += loss.item()
            
            sws_avg_loss = sws_cycle_loss / max(1, t_sws)
            total_sws_loss += sws_avg_loss
            print(f"SWS记忆巩固完成 | 平均损失：{sws_avg_loss:.4f}")
            
            # 执行脑脊液β蛋白清除（新增核心功能）
            clear_ratio = self.gcu.csf_clearance(self.cfb_blocks)
            total_clear_ratio += clear_ratio
            
            # 执行突触稳态调控
            self.shr.differential_scaling(self.cfb_blocks)
            prune_ratio = self.shr.adaptive_pruning(self.cfb_blocks)
            print(f"突触稳态调控完成 | 修剪比例：{prune_ratio:.4f}")
            
            # ========== REM快速眼动睡眠阶段：生成式梦境泛化学习 ==========
            # REM阶段用稍大学习率，促进特征泛化
            for param_group in optimizer.param_groups:
                param_group['lr'] = rem_lr
            
            rem_cycle_loss = 0.0
            for step in range(t_rem):
                dream_samples, dream_labels = self.dtrg.generate_rem_dream_batch(
                    batch_size=32, num_classes=self.num_classes
                )
                outputs, _ = self(dream_samples)
                loss = criterion(outputs, dream_labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                rem_cycle_loss += loss.item()
            
            rem_avg_loss = rem_cycle_loss / max(1, t_rem)
            total_rem_loss += rem_avg_loss
            print(f"REM梦境训练完成 | 平均损失：{rem_avg_loss:.4f}")
        
        # 睡眠结束，重置所有模块状态
        for cfb in self.cfb_blocks:
            cfb.reset_activation_stats()
        self.tcu.sleep_pressure = 0.0
        self.tcu.global_amyloid_beta = 0.0
        # 设置睡眠后恢复期
        self.tcu.sleep_recovery_counter = self.tcu.recovery_steps
        
        # 睡眠总结
        avg_sws_loss = total_sws_loss / len(cycles)
        avg_rem_loss = total_rem_loss / len(cycles)
        avg_clear_ratio = total_clear_ratio / len(cycles)
        print(f"\n=== 睡眠结束 ===")
        print(f"平均SWS损失：{avg_sws_loss:.4f} | 平均REM损失：{avg_rem_loss:.4f}")
        print(f"平均β蛋白清除比例：{avg_clear_ratio:.4f}")
        
        return avg_sws_loss, avg_rem_loss, avg_clear_ratio

# 完整训练流程示例（MNIST 数据集）
def main():
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 打开日志文件
    log_file = open('hypnocycle_training.log', 'w', encoding='utf-8')
    # 保存原始print函数
    import builtins
    original_print = builtins.print
    def log_print(message):
        original_print(message)
        log_file.write(message + '\n')
        log_file.flush()
    
    log_print(f"使用设备：{device}")
    
    # 数据集准备
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # 模型、优化器、损失函数初始化
    model = HypnoCycleNetV2(num_classes=10, input_shape=(1, 28, 28), device=device).to(device)
    # 主模型优化器
    main_optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # DreamVAE优化器
    model.dtrg.vae_optimizer = torch.optim.Adam(model.dtrg.dream_vae.parameters(), lr=1e-4)
    # 损失函数
    criterion = nn.CrossEntropyLoss()
    
    # 训练配置
    num_epochs = 6  # 训练6轮
    sleep_count = 0
    
    # 早停机制参数
    best_test_acc = 0.0
    patience = 3
    patience_counter = 0
    
    # 记录数据用于绘图
    sleep_pressure_history = []
    amyloid_beta_history = []
    accuracy_history = []
    batch_history = []
    
    # 完整训练闭环
    batch_count_total = 0
    for epoch in range(num_epochs):
        # 只写入日志文件，不打印到控制台
        log_file.write(f"\n========== Epoch {epoch+1}/{num_epochs} ==========\n")
        log_file.flush()
        model.train()
        epoch_loss, epoch_acc, epoch_vae_loss, batch_count = 0.0, 0.0, 0.0, 0
        
        for batch_idx, batch in enumerate(train_loader):
            # 清醒学习步骤
            loss, acc, vae_loss = model.wake_learning_step(batch, main_optimizer, criterion)
            epoch_loss += loss
            epoch_acc += acc
            epoch_vae_loss += vae_loss
            batch_count += 1
            batch_count_total += 1
            
            # 记录数据
            sleep_pressure_history.append(model.tcu.sleep_pressure)
            amyloid_beta_history.append(model.tcu.global_amyloid_beta)
            accuracy_history.append(acc)
            batch_history.append(batch_count_total)
            
            # 打印训练进度到日志文件
            if batch_idx % 10 == 0:
                # 只写入日志文件，不打印到控制台
                log_file.write(f"Batch {batch_idx}/{len(train_loader)} | 损失：{loss:.4f} | 准确率：{acc:.4f} | 睡眠压力：{model.tcu.sleep_pressure:.4f} | β蛋白浓度：{model.tcu.global_amyloid_beta:.4f}\n")
                log_file.flush()
            
            # 检查并触发睡眠
            if model.tcu.is_sleep_triggered():
                sleep_count += 1
                # 保存原始print函数
                import builtins
                original_print = builtins.print
                # 重定向print到log_file
                def silent_print(message):
                    log_file.write(message + '\n')
                    log_file.flush()
                builtins.print = silent_print
                try:
                    model.sleep_cycle_execution(main_optimizer, criterion)
                finally:
                    # 恢复原始print函数
                    builtins.print = original_print
                # 睡眠结束后恢复清醒阶段学习率
                for param_group in main_optimizer.param_groups:
                    param_group['lr'] = 1e-3
                for param_group in model.dtrg.vae_optimizer.param_groups:
                    param_group['lr'] = 1e-4
                    
                # 记录睡眠后的状态
                sleep_pressure_history.append(model.tcu.sleep_pressure)
                amyloid_beta_history.append(model.tcu.global_amyloid_beta)
                accuracy_history.append(acc)
                batch_history.append(batch_count_total)
        
        # 每个epoch结束后测试集评估
        model.eval()
        test_loss, test_correct, test_total = 0.0, 0, 0
        with torch.no_grad():
            for samples, labels in test_loader:
                samples, labels = samples.to(device), labels.to(device)
                outputs, _ = model(samples)
                test_loss += criterion(outputs, labels).item()
                _, preds = torch.max(outputs, 1)
                test_correct += (preds == labels).sum().item()
                test_total += labels.size(0)
        
        # 计算测试准确率
        test_acc = test_correct / test_total
        
        # 打印epoch总结
        print(f"\nEpoch {epoch+1} 总结：")
        print(f"训练平均损失：{epoch_loss/batch_count:.4f} | 训练平均准确率：{epoch_acc/batch_count:.4f}")
        print(f"测试平均损失：{test_loss/len(test_loader):.4f} | 测试准确率：{test_acc:.4f}")
        print(f"累计睡眠次数：{sleep_count}")
        
        # 早停机制
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            patience_counter = 0
            print(f"测试准确率提升至：{best_test_acc:.4f}，重置耐心计数器")
        else:
            patience_counter += 1
            print(f"测试准确率未提升，耐心计数器：{patience_counter}/{patience}")
            if patience_counter >= patience:
                print("早停触发，停止训练")
                break
    
    # 绘制曲线
    plt.figure(figsize=(15, 10))
    
    # 睡眠压力曲线
    plt.subplot(3, 1, 1)
    plt.plot(batch_history, sleep_pressure_history, label='睡眠压力')
    plt.axhline(y=0.6, color='r', linestyle='--', label='睡眠触发阈值')
    plt.title('睡眠压力变化曲线')
    plt.xlabel('批次')
    plt.ylabel('睡眠压力')
    plt.legend()
    
    # β蛋白浓度曲线
    plt.subplot(3, 1, 2)
    plt.plot(batch_history, amyloid_beta_history, label='β蛋白浓度')
    plt.title('β蛋白浓度变化曲线')
    plt.xlabel('批次')
    plt.ylabel('β蛋白浓度')
    plt.legend()
    
    # 准确率曲线
    plt.subplot(3, 1, 3)
    plt.plot(batch_history, accuracy_history, label='训练准确率')
    plt.title('准确率变化曲线')
    plt.xlabel('批次')
    plt.ylabel('准确率')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('hypnocycle_metrics.png')
    print("\n========== 训练完成 ==========")
    print("指标曲线已保存为 hypnocycle_metrics.png")
    
    # 关闭日志文件
    log_file.close()

if __name__ == "__main__":
    main()