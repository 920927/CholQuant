# coding: utf-8

import math
import argparse

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import LogLocator

import transformers
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer
import os
import pickle

# ─────────────────────────────────────────────────────────────────
# 1. 模型加载（与你的代码保持一致）
# ─────────────────────────────────────────────────────────────────

def get_llama(model_path):
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    if "Llama-3" in model_path:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    else:
        try:
            tokenizer = LlamaTokenizer.from_pretrained(
                model_path, device_map="cpu", trust_remote_code=True)
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(
                model_path, device_map="cpu", trust_remote_code=True)
    model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype="auto")
    model.seqlen = 2048
    return model, tokenizer


# ─────────────────────────────────────────────────────────────────
# 2. 数据加载（复用你的 datautils）
# ─────────────────────────────────────────────────────────────────

def get_loaders(name, nsamples=128, seed=42, seqlen=2048, model=""):
    """
    直接复用你项目里的 datautils.get_loaders。
    如果找不到，这里提供一个最简的 wikitext2 实现作为后备。
    """
    try:
        from datautils import get_loaders as _get_loaders
        return _get_loaders(name, nsamples=nsamples, seed=seed,
                            model=model, seqlen=seqlen)
    except ImportError:
        pass

    # ── 后备实现（仅支持 wikitext2）────────────────────────────
    from datasets import load_dataset
    from transformers import AutoTokenizer
    import random

    random.seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    data = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join(data["text"])
    enc  = tokenizer(text, return_tensors="pt")

    samples = []
    total_len = enc.input_ids.shape[1]
    for _ in range(nsamples):
        i = random.randint(0, total_len - seqlen - 1)
        inp = enc.input_ids[:, i : i + seqlen]
        tar = inp.clone()
        tar[:, :-1] = -100
        samples.append((inp, tar))
    return samples, samples


# ─────────────────────────────────────────────────────────────────
# 3. Hessian 抽取：在目标层的两个子层上挂 hook
# ─────────────────────────────────────────────────────────────────

class HessianCollector:
    """与你的 GPTQ.add_batch 完全等价，只收集 H，不做量化。"""

    def __init__(self, layer):
        self.layer    = layer
        self.dev      = layer.weight.device
        self.columns  = layer.weight.shape[1]
        if isinstance(layer, nn.Conv2d):
            self.columns = layer.weight.flatten(1).shape[1]
        self.H        = torch.zeros((self.columns, self.columns),
                                    device=self.dev)
        self.nsamples = 0

    def add_batch(self, inp):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, (nn.Linear, transformers.Conv1D)):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        self.H        *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp            = math.sqrt(2 / self.nsamples) * inp.float()
        self.H        += inp.matmul(inp.t())

    def get_diag(self):
        return torch.diag(self.H).cpu().numpy()



@torch.no_grad()
def extract_all_sublayers_hessian(model, dataloader, dev):
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    # --- 1. 捕获初始输入 (逻辑保持不变) ---
    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)
    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((len(dataloader), model.seqlen, model.config.hidden_size), dtype=dtype, device=dev)
    cache = {"i": 0, "attention_mask": None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs.get("attention_mask")
            cache["position_ids"] = kwargs.get("position_ids")
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try: model(batch[0].to(dev))
        except ValueError: pass
    layers[0] = layers[0].module.cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    attention_mask = cache["attention_mask"]
    position_ids = cache["position_ids"]
    outs = torch.zeros_like(inps)

    # --- 2. 遍历所有层并收集所有 Linear 子层 ---
    all_results = {}

    for i in range(len(layers)):
        layer = layers[i].to(dev)
        
        # 自动获取当前层中所有的 Linear 层名和模块
        # 这将包括 q, k, v, o_proj 以及 gate, up, down_proj
        sublayers = {}
        for name, module in layer.named_modules():
            if isinstance(module, nn.Linear):
                sublayers[name] = module

        collectors = {name: HessianCollector(mod) for name, mod in sublayers.items()}

        handles = []
        for name, mod in sublayers.items():
            def make_hook(n):
                def hook(_, inp, __):
                    # 注意：对于大多数层，输入是 inp[0]
                    collectors[n].add_batch(inp[0].data)
                return hook
            handles.append(mod.register_forward_hook(make_hook(name)))

        # 前向传播
        for j in range(inps.shape[0]):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        for h in handles: h.remove()
        
        # 保存这一层所有子场的结果
        all_results[i] = {name: col.get_diag() for name, col in collectors.items()}

        # 内存清理
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps
        print(f"Layer {i} fully processed (Extracted {len(sublayers)} sublayers).")

    model.config.use_cache = use_cache
    return all_results



# ─────────────────────────────────────────────────────────────────
# 4. 绘图
# ─────────────────────────────────────────────────────────────────

LAYER_META = {
    "self_attn.q_proj": {
        "label": r"Attention projection (self_attn.q_proj)",
        "color": "#4C72B0",
        "type":  "Peak-concentrated",
    },
    "mlp.down_proj": {
        "label": r"SiLU-gated MLP (mlp.down_proj)",
        "color": "#C44E52",
        "type":  "Sparsity-induced",
    },
}
def plot_all_sublayers(layer_diags: dict, layer_idx: int, output_path: str):
    """
    layer_diags: { "self_attn.q_proj": array, "mlp.down_proj": array, ... }
    """
    names = sorted(layer_diags.keys())
    num_subs = len(names)
    cols = 4
    rows = (num_subs + cols - 1) // cols

    fig = plt.figure(figsize=(20, 4 * rows))
    gs = gridspec.GridSpec(rows, cols, figure=fig, wspace=0.3, hspace=0.4)

    for idx, name in enumerate(names):
        diag = layer_diags[name]
        ax = fig.add_subplot(gs[idx // cols, idx % cols])
        
        # 数据处理
        dead_mask = diag <= 1e-12 # 稍微提高阈值处理浮点误差
        valid = diag[~dead_mask]
        
        if len(valid) > 0:
            x = np.arange(len(diag))
            ax.scatter(x[~dead_mask], diag[~dead_mask], s=0.5, alpha=0.5, color="#4C72B0", rasterized=True)
            
            # 计算指标
            kappa = valid.max() / valid.min()
            log_kappa = np.log10(kappa)
            
            # 标注
            ax.set_yscale("log")
            ax.set_title(f"{name}\n" + r"$\kappa \approx 10^{" + f"{log_kappa:.1f}" + r"}$", fontsize=10)
        if dead_mask.any():
            ax.text(0.05, 0.05, f"Dead: {dead_mask.mean():.1%}", transform=ax.transAxes, color="red", fontsize=8)

        ax.tick_params(labelsize=8)
        ax.set_xlabel("Index", fontsize=8)

    fig.suptitle(f"Hessian Diagonals - Layer {layer_idx}", fontsize=14, y=1.02)
    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close()
    
    
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import seaborn as sns

def plot_all_sublayers_paper_ready(layer_diags: dict, layer_idx: int, output_path: str):
    """
    针对顶会论文优化的 Hessian Diagonals 可视化函数
    """
    # 设置全局绘图风格
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "text.usetex": False,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        # 添加以下三行设置坐标轴为黑色
        "axes.edgecolor": "black",    # 坐标轴颜色
        "xtick.color": "black",       # x轴刻度颜色
        "ytick.color": "black",       # y轴刻度颜色
    })
    
    names = sorted(layer_diags.keys())
    num_subs = len(names)
    cols = 4
    rows = (num_subs + cols - 1) // cols

    # 调整比例，确保子图不会太挤
    fig = plt.figure(figsize=(16, 3.5 * rows))
    gs = gridspec.GridSpec(rows, cols, figure=fig, wspace=0.35, hspace=0.5)

    # 调色盘：深蓝色用于数据点
    main_color = "#2c7fb8"
    accent_color = "#d95f02" # 用于高亮或异常

    for idx, name in enumerate(names):
        diag = layer_diags[name]
        ax = fig.add_subplot(gs[idx // cols, idx % cols])
        
        # 数据清理
        threshold = 1e-12
        dead_mask = diag <= threshold
        valid = diag[~dead_mask]
        
        if len(valid) > 0:
            x = np.arange(len(diag))
            # 使用更小的 alpha 和点大小，并增加 rasterized 处理大数据点
            ax.scatter(x[~dead_mask], diag[~dead_mask], 
                       s=1.5, alpha=0.4, color=main_color, 
                       edgecolors='none', rasterized=True)
            
            # 计算 Condition Number (kappa)
            kappa = valid.max() / valid.min()
            log_kappa = np.log10(kappa)
            
            # 科学计数法格式化标题
            # 缩短名称以防溢出 (例如 self_attn.q_proj -> q_proj)
            short_name = name.split('.')[-1] if '.' in name else name
            ax.set_title(fr"{short_name}" + "\n" + fr"$\log_{{10}}(\kappa) \approx {log_kappa:.1f}$", fontweight="bold", fontsize=16)
            ax.set_yscale("log")
        
        # 优化坐标轴
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)

        # 标注 Dead Neurons 比例
        dead_pct = dead_mask.mean()
        if dead_pct > 0:
            text_color = accent_color if dead_pct > 0.1 else "grey"
            ax.text(0.95, 0.05, f"Dead: {dead_pct:.1%}", 
                    transform=ax.transAxes, color=text_color, 
                    fontsize=9, fontweight='bold',
                    ha='right', va='bottom', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

        if idx % cols == 0:
            ax.set_ylabel("Eigenvalue (log)", fontsize=9)
        if idx >= (rows - 1) * cols:
            ax.set_xlabel("Parameter Index", fontsize=9)

    # 总标题优化
    fig.suptitle(f"Hessian Spectral Analysis: Transformer Layer {layer_idx}", 
                 fontsize=14, fontweight="bold", y=0.98)

    # 保存：使用 300 DPI 以满足出版要求
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()
    
    
import numpy as np
import matplotlib.pyplot as plt

def plot_selected_sublayers_paper(layer_diags: dict, layer_idx: int, output_path: str):
    """
    调整说明：
    1. 颜色恢复：使用 #2c7fb8 (主色) 和 #e63946 (死神经元)。
    2. 字体调大：增加 title, label 和 tick 的字号。
    3. 间距缩短：使用 subplots_adjust 精确控制顶部间距。
    """
    # 1. 挑选特定的层
    target_subs = ["self_attn.q_proj", "self_attn.o_proj", "mlp.down_proj"]
    selected_names = [n for n in target_subs if n in layer_diags]
    
    if not selected_names:
        print(f"Warning: No matching sublayers found in Layer {layer_idx}")
        return

    # 2. 设置学术风格 (进一步调大字号)
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "mathtext.fontset": "stix",
        "axes.labelsize": 20,      # 调大轴标签
        "axes.titlesize": 26,      # 调大子图标题
        "xtick.labelsize": 18,     # 调大刻度
        "ytick.labelsize": 18,
    })

    # 3. 创建画布
    fig, axes = plt.subplots(1, len(selected_names), figsize=(18, 5.2))
    if len(selected_names) == 1: axes = [axes]

    # 恢复原定颜色
    main_color = "#2c7fb8"
    dead_color = "#e63946"

    # 总标题：y值下调以缩短间距
#     fig.suptitle(
#         f"Hessian Diagonal Distribution: Layer {layer_idx}",
#         fontsize=22, fontweight='bold', y=0.95
#     )

    for i, name in enumerate(selected_names):
        ax = axes[i]
        diag = layer_diags[name]
        
        # 数据清理
        threshold = 1e-12
        dead_mask = diag <= threshold
        valid = diag[~dead_mask]
        
        # 绘制散点
        x = np.arange(len(diag))
        ax.scatter(x[~dead_mask], diag[~dead_mask], 
                   s=3.5, alpha=0.5, color=main_color, 
                   edgecolors='none', rasterized=True)
        
        # 对数坐标
        ax.set_yscale("log")
        
        if len(valid) > 0:
            ymin, ymax = valid.min(), valid.max()
            ax.set_ylim(ymin * 0.4, ymax * 6.0) # 增加顶部裕量
            
            kappa = ymax / ymin
            log_kappa = np.log10(kappa)
            
            # 子图标题 (加粗且调大)
            title_name = name.split('.')[-1].replace('_', r'\_')
            ax.set_title(fr"$\mathbf{{{title_name}}}$" "  " + fr"$\log_{{10}}(\kappa) \approx {log_kappa:.1f}$", 
                         pad=6)
        
        # 标注死神经元
        dead_pct = dead_mask.mean()
        if dead_pct > 0:
            ax.text(0.95, 0.05, f"Dead: {dead_pct:.1%}", 
                    transform=ax.transAxes, color=dead_color, 
                    fontsize=12, fontweight='bold', ha='right', va='bottom',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

        ax.set_xlabel("Neuron Index")
        ax.set_ylabel("Hessian Diagonal")
        
        # 边框与网格
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, which='both', linestyle=':', linewidth=0.5, alpha=0.6)

    # 4. 极致缩短布局间距
    # top=0.82 强行拉近总标题与子图标题的距离
    plt.subplots_adjust(top=0.82, bottom=0.15, left=0.06, right=0.96, wspace=0.25)
    
    # 保存
    plt.savefig(output_path, bbox_inches="tight", dpi=600)
    plt.close()
    print(f"Plot saved with enlarged font and compact spacing to: {output_path}")

    
# ─────────────────────────────────────────────────────────────────
# 5. 入口
# ─────────────────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="/ruisun2025/dx/llm_model/Llama-2-7b-hf")
    parser.add_argument("--dataset", type=str, default="wikitext2", choices=["wikitext2", "ptb", "c4"])
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    
    parser.add_argument("--extract", action="store_true", help="是否执行模型推理提取数据")
    parser.add_argument("--visualize", action="store_true", help="是否根据已保存的数据生成图表")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    model_name = args.model.split('/')[-1]
    data_file = f"{model_name}_{args.dataset}_hessian.pkl"
    output_dir = "all_layers_analysis"

    if args.extract:
        DEV = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"开始提取数据，模型: {args.model}")
        
        model, _ = get_llama(args.model)
        model = model.eval()
        dataloader, _ = get_loaders(args.dataset, nsamples=args.nsamples, 
                                   seed=args.seed, model=args.model, seqlen=model.seqlen)

        all_data = extract_all_sublayers_hessian(model, dataloader, DEV)
        
        # 将数据序列化到磁盘
        with open(data_file, 'wb') as f:
            pickle.dump(all_data, f)
        print(f"数据提取完成，已保存至: {data_file}")

    if args.visualize:
        if not os.path.exists(data_file):
            print(f"错误: 找不到数据文件 {data_file}。请先运行 --extract")
            return

        print(f"正在从 {data_file} 加载数据并生成图表...")
        with open(data_file, 'rb') as f:
            all_data = pickle.load(f)

        os.makedirs(output_dir, exist_ok=True)
        
        for l_idx, sub_data in all_data.items():
            out_p = f"{output_dir}/layer_{l_idx}.png"
#             plot_all_sublayers(sub_data, l_idx, out_p)
            plot_all_sublayers_paper_ready(sub_data, l_idx, out_p)
            print(f"已生成可视化: {out_p}")

            if l_idx == 0:
                plot_selected_sublayers_paper(sub_data, 0, "layer0_comparison.pdf")
            
if __name__ == "__main__":
    main()