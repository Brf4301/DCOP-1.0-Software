import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from rdkit import Chem
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, global_mean_pool
import torch.nn.functional as F
from torch.nn.functional import cosine_similarity
import matplotlib.pyplot as plt
import numpy as np
import gc
from torch.cuda.amp import GradScaler, autocast
import time
import urllib.request
# from gnn_sp import GNN, temperature
import os


# 解析MGF文件，包括MS2信息
def load_mgf_with_ms2(filepath):
    data = []
    current_spectrum = {}
    ms2_data = []

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith("NAME="):
                if current_spectrum:
                    ms2_data = [(mz, intensity) for mz, intensity in ms2_data if intensity > 100]
                    current_spectrum["MS2"] = ms2_data
                    data.append(current_spectrum)
                current_spectrum = {"name": line[5:]}
                ms2_data = []
            elif line.startswith("SMILES="):
                current_spectrum["SMILES"] = line[7:]
            elif line.startswith("PEPMASS="):
                current_spectrum["PEPMASS"] = line[8:]
            elif line.startswith("CHARGE="):
                current_spectrum["CHARGE"] = line[7:]
            elif line and line[0].isdigit():
                mz, intensity = map(float, line.split())
                ms2_data.append((mz, intensity))

    if current_spectrum:
        ms2_data = [(mz, intensity) for mz, intensity in ms2_data if intensity > 100]
        current_spectrum["MS2"] = ms2_data
        data.append(current_spectrum)

    return data

# 将分子结构转换为图数据（更新原子特征元素）
def mol_to_graph(mol):
    node_features = []
    for atom in mol.GetAtoms():
        features = [0] * 12      # C/N/O/F/S/Cl/Na/Al/P/Br/K/Mg
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
    edge_features = []
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
        edge_indices.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
        edge_features.append(bond_features)

    edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    node_features = torch.tensor(node_features, dtype=torch.float32)
    edge_attr = torch.tensor(edge_features, dtype=torch.float32)
    return edge_index, node_features, edge_attr

# 将MS2谱图数据转换为向量
def ms2_to_vector(ms2_data, mz_range=(74, 400), bins=326000, intensity_transform="sqrt"):
    bin_size = (mz_range[1] - mz_range[0]) / bins
    vector = np.zeros(bins, dtype=np.float32)
    # 选取强度最高的200个峰
    ms2_data = sorted(ms2_data, key=lambda x: x[1], reverse=True)[:200]

    for mz, intensity in ms2_data:
        if mz_range[0] <= mz < mz_range[1]:
            bin_idx = int((mz - mz_range[0]) / bin_size)
            if intensity_transform == "sqrt":
                intensity = np.sqrt(intensity)
            vector[bin_idx] += intensity

    total_intensity = vector.sum()
    if total_intensity > 0:
        vector /= total_intensity  # 归一化
    else:
        vector = np.zeros_like(vector)
    return torch.tensor(vector, dtype=torch.float32).unsqueeze(0)

# 改进的MPNN模型
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

# 数据加载和划分
data = load_mgf_with_ms2('./MASSBANK.mgf')

# 分批处理
batch_size = 200  # 每次处理 100 条数据
for i in range(0, len(data), batch_size):
    batch_data = data[i:i + batch_size]

mols = []
for item in data:
    smiles = item.get('SMILES')
    ms2_data = item.get('MS2')
    if smiles and ms2_data:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            edge_index, node_features, edge_attr = mol_to_graph(mol)
            ms2_vector = ms2_to_vector(ms2_data)
            mols.append(Data(
                x=node_features,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=ms2_vector
            ))

print(f"Loaded {len(mols)} molecules with MS2 data.")

# 定义设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 数据划分
random.shuffle(mols)
train_size = int(0.8 * len(mols))
val_size = int(0.1 * len(mols))
test_size = len(mols) - train_size - val_size
train_mols, val_mols, test_mols = mols[:], mols[train_size:train_size + val_size], mols[train_size + val_size:]
print(train_size, val_size, test_size)

train_loader = DataLoader(train_mols, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_mols, batch_size=32, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_mols, batch_size=32, num_workers=4, pin_memory=True)
print(len(train_loader), len(val_loader), len(test_loader))

# 模型初始化
model = MPNNWithAttention(in_feats=12, h_feats=256, out_feats=326000, num_layers=3).to(device)
# model = GNN(num_layer=4, input_dim=12, emb_dim=1024, output_dim=326000, JK = "last", drop_ratio = 0, gnn_type = "gin", disable_fingerprint = False).to(device)

# 参数初始化
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

model.apply(init_weights)

# 自定义损失函数
def loss_fn_with_cosine(predicted, target, alpha=0.5):
    criterion_mes = nn.MSELoss()
    target = target.squeeze(1)
    mse_loss = criterion_mes(predicted, target)
    cosine_loss = 1 - cosine_similarity(predicted, target, dim=1).mean()
    return alpha * mse_loss + (1 - alpha) * cosine_loss, 1-cosine_loss, mse_loss

# 优化器和调度器
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.1)

# 初始化学习率调度器
scheduler = ReduceLROnPlateau(optimizer,
                              mode='min',  # 'min' 表示目标是最小化验证损失
                              factor=0.1,  # 学习率每次降低的比例，例如从 0.001 -> 0.0001
                              patience=10, # 容忍验证损失在 10 个 epoch 内不下降
                              verbose=True, # 输出学习率调整信息
                              min_lr=1e-6) # 最低学习率限制

# AMP 初始化
scaler = GradScaler()

# 开始训练和验证循环
num_epochs = 100
train_losses, val_losses = [], []

for epoch in range(1, num_epochs+1):
    # Training Loop
    model.train()
    train_loss = torch.zeros(1).to(device)
    for batch in train_loader:
        batch.x = batch.x.to(device)
        batch.edge_index = batch.edge_index.to(device)
        batch.edge_attr = batch.edge_attr.to(device)
        batch.y = batch.y.to(device).view(batch.num_graphs, -1)

        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch.to(device))
        loss, _, _ = loss_fn_with_cosine(out, batch.y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        train_loss += loss


    train_loss /= len(train_loader)
    train_losses.append(train_loss.item())

    # Validation Loop
    model.eval()
    val_loss = torch.zeros(1).to(device)
    cos_sum = 0.0
    mse_sum = 0.0
    with torch.no_grad():
        for batch in val_loader:
            batch.x = batch.x.to(device)
            batch.edge_index = batch.edge_index.to(device)
            batch.edge_attr = batch.edge_attr.to(device)
            batch.y = batch.y.to(device)

            batch_size = batch.num_graphs
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch.to(device))
            target = batch.y.view(batch_size, -1)
            loss1, cosine_similarities, mse_score = loss_fn_with_cosine(out, target)
            val_loss += loss1
            cos_sum += cosine_similarities.item()
            mse_sum += mse_score.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss.item())
        cos_sim = cos_sum / len(val_loader)
        mse_sum = mse_sum / len(val_loader)

    if (epoch > 1) and (epoch % 10 == 0):
        # 保存检查点
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_losses': train_losses,
            'val_losses': val_losses,
            'mean_cosine_similarity': cos_sim,  # 保存完整序列
        }, f"mpnn_with_base256_{epoch}+.pth")


    # 调用调度器，根据验证损失调整学习率
    scheduler.step(val_loss)

    # 获取当前学习率并打印
    current_lr = optimizer.param_groups[0]['lr']
    print(
        f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss.item():.4f} - Val Loss: {val_loss.item():.4f} "
        f"- LR: {current_lr:.6f} - cos: {cos_sim:.4f} - mse: {mse_sum:.4f}")

    # 如果学习率降到最低值，终止训练
    if current_lr <= scheduler.min_lrs[0]:
        print("Learning rate has reached its minimum value. Stopping early.")
        break

# 绘制损失曲线
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()
plt.savefig('./figs/loss.png', format='png')

# 测试模型性能并计算余弦相似度
model.eval()
test_cosine_similarities = []
test_mse = []
with torch.no_grad():
    for batch in test_loader:
        batch.x = batch.x.to(device)
        batch.edge_index = batch.edge_index.to(device)
        batch.edge_attr = batch.edge_attr.to(device)
        batch.y = batch.y.to(device)

        batch_size = batch.num_graphs
        out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch.to(device))
        target = batch.y.view(batch_size, -1)
        loss_total, cos, mse = loss_fn_with_cosine(out, target)
        temperature=5
        cosine_sim = torch.sigmoid(cosine_similarity(out, target, dim=1)*temperature)
        test_cosine_similarities.append(cosine_sim.cpu().numpy())
        test_mse.append(mse.cpu().numpy())

mean_cosine_similarity = np.mean(test_cosine_similarities)
test_mse = np.mean(test_mse)
print(f"Mean Cosine Similarity on Test Set: {mean_cosine_similarity:.4f} - mse: {test_mse}")
x1 = f"Mean Cosine Similarity on Test Set: {mean_cosine_similarity:.4f} - mse: {test_mse}"

# 检查文件是否存在
if os.path.exists("output.txt"):
    # 文件存在，使用追加模式写入
    with open("output.txt", "a", encoding="utf-8") as file:
        file.write(x1)
    print("文件已存在，内容已追加。")
else:
    # 文件不存在，创建文件并写入
    with open("output.txt", "w", encoding="utf-8") as file:
        file.write(x1)
    print("文件不存在，已创建文件并写入内容。")

def comput_cos(A, B):
    # 计算点积
    dot_product = np.dot(A, B)
    # 计算两个向量的模长
    norm_A = np.linalg.norm(A)
    norm_B = np.linalg.norm(B)
    # 计算余弦相似度
    return dot_product / (norm_A * norm_B)
# 预测结果与真实谱图的对比
model.eval()
with torch.no_grad():
    cos_200 = []
    cos_mz = []
    num = 0
    for batch in test_loader:
        num += 1
        batch.x = batch.x.to(device)
        batch.edge_index = batch.edge_index.to(device)
        batch.edge_attr = batch.edge_attr.to(device)
        batch.y = batch.y.to(device)
        batch_size = batch.num_graphs
        out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch.to(device))
        target = batch.y.view(batch_size, -1)

        mz_range = (74, 400)
        bins = 326000 # 326000
        bin_size = (mz_range[1] - mz_range[0]) / bins
        mz_values = np.linspace(mz_range[0], mz_range[1], bins)

        for i in range(min(5, batch_size)):
            # 预测和真实谱图
            predicted_spectrum = out[i].cpu().numpy()
            target_spectrum = target[i].cpu().numpy()

            # 归一化
            predicted_spectrum /= np.max(np.abs(predicted_spectrum))
            target_spectrum /= np.max(np.abs(target_spectrum))

            # 筛选预测值中响应强度最高的 200 个 m/z 值
            top_indices = np.argsort(predicted_spectrum)[-200:]  # 强度从小到大排序，取最后 200 个
            filtered_mz_values = mz_values[top_indices]
            filtered_predicted_intensities = predicted_spectrum[top_indices]

            # 筛选真实谱图中响应强度最高的 200 个 m/z 值
            top_target_indices = np.argsort(target_spectrum)[-200:]
            filtered_target_mz_values = mz_values[top_target_indices]
            filtered_target_intensities = target_spectrum[top_target_indices]

            cos_200.append(comput_cos(filtered_target_intensities, filtered_predicted_intensities))
            cos_mz.append(comput_cos(filtered_target_mz_values, filtered_mz_values))

    # print('cos_200:', np.mean(cos_200), 'cos_mz:', np.mean(cos_mz))
            # 绘图
            plt.figure(figsize=(10, 6))
            plt.stem(filtered_mz_values, filtered_predicted_intensities, linefmt='b-', markerfmt='bo', label="Predicted Spectrum", basefmt=" ")
            plt.stem(filtered_target_mz_values, -filtered_target_intensities, linefmt='orange', markerfmt='ro', label="Target Spectrum (Inverted)", basefmt=" ")

            plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
            plt.xlabel('m/z')
            plt.ylabel('Normalized Intensity')
            plt.title(f'Sample {i + 1} - Predicted vs Target Spectrum (Top 200 Peaks)')
            plt.legend()
            plt.ylim([-1, 1])
            plt.show()
            plt.savefig(f'./figs/fig{num}{i+1}.png', format='png')
    print('cos_200:', np.mean(cos_200), 'cos_mz:', np.mean(cos_mz))
    x2 = f"cos_200: {np.mean(cos_200)}, cos_mz: {np.mean(cos_mz)}"
    if os.path.exists("output.txt"):
        # 文件存在，使用追加模式写入
        with open("output.txt", "a", encoding="utf-8") as file:
            file.write(x2)
        print("文件已存在，内容已追加。")
    else:
        # 文件不存在，创建文件并写入
        with open("output.txt", "w", encoding="utf-8") as file:
            file.write(x2)
        print("文件不存在，已创建文件并写入内容。")
