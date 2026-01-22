# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 14:43:15 2025

@author: Huawei
"""

from torch_geometric.nn import GATConv, global_mean_pool
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch
import numpy as np
import torch.nn as nn
# from gnn_sp import GNN
from torch.optim.lr_scheduler import ReduceLROnPlateau
print('0000')
num = 20

# 定义模型
class MPNNWithAttention(nn.Module):
    def __init__(self, in_feats, h_feats=64, out_feats=326000, num_layers=4):
        super(MPNNWithAttention, self).__init__()
        self.num_layers = num_layers
        self.node_embedding = nn.Linear(in_feats, h_feats)
        self.mpnn_layers = nn.ModuleList(
            [GATConv(h_feats, h_feats, heads=4, concat=False),
             GATConv(h_feats, h_feats, heads=4, concat=False),
             GATConv(h_feats, h_feats, heads=4, concat=False)]
        )
        self.fc = nn.Linear(h_feats, out_feats)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, edge_index, edge_attr, batch):
        h = self.node_embedding(x)
        for layer in self.mpnn_layers:
            # print(h.size())
            h = F.relu(layer(h, edge_index))
        h = global_mean_pool(h, batch)
        h = self.dropout(h)
        return self.fc(h)

# 选择设备，GPU优先
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# 初始化模型和优化器
model = MPNNWithAttention(in_feats=12, h_feats=256, out_feats=326000, num_layers=3).to(device)
# model = GNN(num_layer=6, input_dim=12, emb_dim=1024, output_dim=326000, JK = "last", drop_ratio = 0, gnn_type = "gin", disable_fingerprint = False).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = ReduceLROnPlateau(optimizer,
                              mode='min',  # 'min' 表示目标是最小化验证损失
                              factor=0.1,  # 学习率每次降低的比例，例如从 0.001 -> 0.0001
                              patience=10, # 容忍验证损失在 10 个 epoch 内不下降
                              verbose=True, # 输出学习率调整信息
                              min_lr=1e-6) # 最低学习率限制

# 初始化变量
train_losses = []
val_losses = []
cosine_similarities = []
epoch = 0

# 加载模型的状态
try:
    checkpoint = torch.load(f"E:/博士课题/含P化合物/质谱信息/code/小模型/mpnn_with_base256_20+.pth", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    train_losses = checkpoint['train_losses']
    val_losses = checkpoint['val_losses']
    epoch = checkpoint['epoch']

    # 修复 cosine_similarities 的问题
    cosine_similarities = checkpoint.get('mean_cosine_similarity', [])
    if isinstance(cosine_similarities, (float, np.float32)):  # 如果是标量
        cosine_similarities = [cosine_similarities]  # 转为列表
    elif not isinstance(cosine_similarities, list):
        raise ValueError("cosine_similarities 应该是一个列表或数组类型")

    print(f"模型已从 'mpnn_with_base256_{num}+.pth' 加载，恢复训练自第 {epoch + 1} 轮")
except FileNotFoundError:
    print("未找到检查点文件，初始化新的训练状态")
except KeyError as e:
    print(f"加载检查点时发生错误：缺少键 {e}")

# 可视化余弦相似度随训练轮次的变化
if len(cosine_similarities) > 1:
    plt.plot(range(len(cosine_similarities)), cosine_similarities, label='Cosine Similarity')
    plt.xlabel('Epoch')
    plt.ylabel('Cosine Similarity')
    plt.title('Training Cosine Similarity Over Epochs')
    plt.legend()
    plt.show()
else:
    print(f"cosine_similarities 数据不足以绘图: {cosine_similarities}")

# 打印模型、优化器和调度器的状态
print(model)
print(optimizer)
print(scheduler)
print(f"当前调度器状态: {scheduler.state_dict()}")


# 示例训练和保存逻辑
def train_model(num_epochs):
    global epoch, train_losses, val_losses, cosine_similarities

    for epoch in range(epoch, num_epochs):
        # 模拟训练逻辑
        train_loss = np.random.random()  # 模拟训练损失
        val_loss = np.random.random()  # 模拟验证损失
        cosine_similarity = np.random.random()  # 模拟余弦相似度

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        cosine_similarities.append(cosine_similarity)

        # 调整学习率
        scheduler.step(val_loss)

        # 保存检查点
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_losses': train_losses,
            'val_losses': val_losses,
            'mean_cosine_similarity': cosine_similarities,  # 保存完整序列
        }, "mpnn_with_base256_{num}+.pth")

        print(
            f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Cosine Similarity: {cosine_similarity:.4f}")


# 示例调用训练
#train_model(num_epochs=5)


import pandas as pd
import numpy as np
import torch
from rdkit import Chem
from torch_geometric.data import Data

mz_range = (74, 400)  # 根据实际情况调整范围

# 定义 mol_to_graph 函数
def mol_to_graph(mol):
    node_features = []
    for atom in mol.GetAtoms():
        features = [0] * 12  # 新的元素集合：C, N, O, F, S, Cl, Na, Al, P, Br, K, Mg
        atomic_num = atom.GetAtomicNum()
        # 根据元素的原子序号设置对应特征
        if atomic_num == 6:  # C
            features[0] = 1
        elif atomic_num == 7:  # N
            features[1] = 1
        elif atomic_num == 8:  # O
            features[2] = 1
        elif atomic_num == 9:  # F
            features[3] = 1
        elif atomic_num == 16:  # S
            features[4] = 1
        elif atomic_num == 17:  # Cl
            features[5] = 1
        elif atomic_num == 11:  # Na
            features[6] = 1
        elif atomic_num == 13:  # Al
            features[7] = 1
        elif atomic_num == 15:  # P
            features[8] = 1
        elif atomic_num == 35:  # Br
            features[9] = 1
        elif atomic_num == 19:  # K
            features[10] = 1
        elif atomic_num == 12:  # Mg
            features[11] = 1
        node_features.append(features)

    # 构建边特征
    edge_indices = []
    edge_attr = []  # 初始化 edge_attr

    for bond in mol.GetBonds():
        bond_type = bond.GetBondTypeAsDouble()
        bond_features = [0] * 6  # 单键/双键/三键/芳香键/环状键/其他
        if bond_type == 1.0:
            bond_features[0] = 1
        elif bond_type == 2.0:
            bond_features[1] = 1
        elif bond_type == 3.0:
            bond_features[2] = 1
        if bond.GetIsAromatic():
            bond_features[3] = 1
        if bond.IsInRing():
            bond_features[4] = 1
        edge_indices.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
        edge_indices.append([bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()])
        edge_attr.append(bond_features)
        edge_attr.append(bond_features)  # 无向图需要双向边

    # 转换 edge_index 和 edge_attr 为 PyTorch 张量
    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    # 转换节点特征为 PyTorch 张量
    node_features = torch.tensor(node_features, dtype=torch.float)

    return edge_index, node_features, edge_attr



# 定义 CSV 文件路径
input_csv_path = 'E:/博士课题/含P化合物/质谱信息/OPEs/OPEs_predict.csv'  # 替换为你的 CSV 路径
output_csv_path = 'E:/博士课题/含P化合物/质谱信息/OPEs/results_base256_20+.csv'

# 读取 CSV 文件
df = pd.read_csv(input_csv_path)

# 检查是否有 'SMILES' 列
if 'SMILES' not in df.columns:
    raise ValueError("The CSV file必须包含 'SMILES' 列。")


# 预测结果存储
results = []

# 模型评估模式
model.eval()

# 模型预测逻辑修正
# 模型预测逻辑修正
with torch.no_grad():
    for index, row in df.iterrows():
        smiles = str(row['SMILES'])  # 确保 SMILES 是字符串
        compound_id = str(row.get('id', index))  # 确保 id 是字符串
        # 转换为分子对象
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            print(f"Invalid SMILES for row {index}: {smiles}")
            continue
        # 转换为图数据
        edge_index, node_features, edge_attr = mol_to_graph(mol)
        node_features = node_features.to(device)
        edge_index = edge_index.to(device)
        edge_attr = edge_attr.to(device)
        # 创建 Data 对象
        graph_data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr
        )
        # 模型预测
        graph_data = graph_data.to(device)
        out = model(
            graph_data.x,
            graph_data.edge_index,
            graph_data.edge_attr,
            torch.zeros(graph_data.x.shape[0], dtype=torch.long, device=device)  # 假设 batch=0
        )
        # 检查输出形状
        print(f"Output shape: {out.shape}")
        predicted_spectrum = out.squeeze().cpu().numpy()  # 去掉 batch 维度
        bins = 326000  # 调整为较小的分箱数326000
        mz_values = np.linspace(mz_range[0], mz_range[1], bins)
        if len(predicted_spectrum) != bins:
            raise ValueError(f"预测谱长度 ({len(predicted_spectrum)}) 和频谱分箱数 ({bins}) 不匹配！")
        # 获取强度最高的前 100 个峰
        top_indices = np.argsort(predicted_spectrum)[-100:]
        filtered_mz_values = mz_values[top_indices]
        filtered_intensities = predicted_spectrum[top_indices]
        # 将 m/z 和 intensity 展平到行
        result_row = {'id': compound_id, 'SMILES': smiles}
        for i, (mz, intensity) in enumerate(zip(filtered_mz_values, filtered_intensities)):
            result_row[f'mz_{i+1}'] = mz
            result_row[f'intensity_{i+1}'] = intensity
        results.append(result_row)

# 将结果转换为 DataFrame 并保存
results_df = pd.DataFrame(results)
results_df = results_df.astype(str)  # 确保所有列为字符串类型
results_df.to_csv(output_csv_path, index=False)



print(f"Predicted MS2 spectra saved to {output_csv_path}")
