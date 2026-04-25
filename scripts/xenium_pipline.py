from pathlib import Path
import pandas as pd
import scanpy as sc
import numpy as np
import os
import gc
import matplotlib.pyplot as plt
import seaborn as sns
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULT_DIR = ROOT / "results"
FIG_DIR = RESULT_DIR / "figures"
PROC_DIR = RESULT_DIR / "processed"
for folder in [FIG_DIR, PROC_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# 定义一个获取路径的通用函数
def get_path(sample_id, type="data"):
    if type == "data":
        return DATA_DIR / f"{sample_id}.h5ad"
    elif type == "figure":
        return FIG_DIR / f"{sample_id}"  # 返回文件夹路径
    return PROC_DIR / f"{sample_id}_processed.h5ad"

def load_xenium_data(data_dir):
    """
    通过读取 h5 矩阵和 cells.csv 手动构建 AnnData
    """
    print(f"--- 开始加载样本: {os.path.basename(data_dir)} ---")
    
    # 1. 路径检查
    h5_path = os.path.join(data_dir, "cell_feature_matrix.h5")
    csv_path = os.path.join(data_dir, "cells.csv.gz")
    
    if not os.path.exists(h5_path) or not os.path.exists(csv_path):
        raise FileNotFoundError(f"在目录中找不到必要的 h5 或 csv 文件，请检查路径。")

    # 2. 读取基因表达矩阵 (X)
    print("正在读取 cell_feature_matrix.h5...")
    adata = sc.read_10x_h5(h5_path)
    # 确保索引是字符串，方便后续 join
    adata.obs_names = adata.obs_names.astype(str)
    
    # 3. 读取细胞元数据与坐标 (obs)
    print("正在读取 cells.csv.gz...")
    cells_df = pd.read_csv(csv_path)
    cells_df['cell_id'] = cells_df['cell_id'].astype(str)
    cells_df.set_index('cell_id', inplace=True)
    
    # 4. 合并数据 (只取两个文件中共有的细胞)
    print(f"矩阵细胞数: {adata.n_obs}, CSV细胞数: {len(cells_df)}")
    # 使用 join 将坐标信息合并进 adata.obs
    adata.obs = adata.obs.join(cells_df, how='inner')
    print(f"对齐后细胞数: {adata.n_obs}")

    # 5. 设置空间坐标槽位 (obsm)
    # 查找 Xenium 标准坐标列名
    if 'x_centroid' in adata.obs.columns and 'y_centroid' in adata.obs.columns:
        adata.obsm['spatial'] = adata.obs[['x_centroid', 'y_centroid']].values
    else:
        print("警告: 未在 CSV 中找到 x_centroid/y_centroid 列！")

    # 6. 基础清洗：移除控制探针 (Xenium 包含 Blank/Negative 控制点)
    # 这些点在聚类分析时通常需要移除
    print("正在过滤控制探针...")
    is_gene = ~adata.var_names.str.startswith(('BLANK', 'NegControl', 'Unassigned', 'Deprecated'))
    adata = adata[:, is_gene].copy()
    
    print(f"✓ 加载完成! 当前数据包含 {adata.n_obs} 个细胞和 {adata.n_vars} 个基因。")
    return adata






### 进行质控和生成数据报告
def basic_qc_filtering(adata: sc.AnnData, 
                      min_reads: int = 10,
                      min_genes: int = 5,
                      qv_threshold: float = 20) -> sc.AnnData:
    """
    基础质控过滤（基于 Nature Methods 2025 标准）
    
    Parameters
    ----------
    adata : sc.AnnData
        输入数据
    min_reads : int
        最小 reads 数阈值（Nature Methods 2025 推荐：<10 的细胞排除）
    min_genes : int
        最小检测基因数
    qv_threshold : float
        质量值阈值
        
    Returns
    -------
    adata : sc.AnnData
        过滤后的数据
    """
    
    print("\n" + "=" * 60)
    print("Basic QC Filtering (Nature Methods 2025 Standard)")
    print("=" * 60)
    
    # 1. 计算基础 QC 指标
    # 使用 .A1 确保将矩阵求和结果转为 1D 数组
    adata.obs['n_counts'] = adata.X.sum(axis=1).A1 if hasattr(adata.X, 'A1') else np.array(adata.X.sum(axis=1)).flatten()
    adata.obs['n_genes'] = (adata.X > 0).sum(axis=1).A1 if hasattr(adata.X, 'A1') else np.array((adata.X > 0).sum(axis=1)).flatten()
    
    # 2. Nature Methods 2025 关键标准：<10 assigned reads 的细胞排除
    print(f"\nFiltering cells with <{min_reads} assigned reads...")
    initial_cells = adata.n_obs
    adata = adata[adata.obs['n_counts'] >= min_reads, :].copy() # 建议加上 .copy() 避免 View 警告
    filtered_cells = initial_cells - adata.n_obs
    print(f" - Removed {filtered_cells:,} cells ({filtered_cells/initial_cells*100:.2f}%)")
    print(f" - Remaining cells: {adata.n_obs:,}")
    
    # 3. 基因过滤
    print(f"\nFiltering genes detected in <{min_genes} cells...")
    initial_genes = adata.n_vars
    sc.pp.filter_genes(adata, min_cells=min_genes)
    filtered_genes = initial_genes - adata.n_vars
    print(f" - Removed {filtered_genes:,} genes")
    print(f" - Remaining genes: {adata.n_vars:,}")
    
    # 4. QC 指标统计
    print("\n" + "-" * 60)
    print("QC Metrics Summary:")
    print("-" * 60)
    print(f" Total counts per cell:")
    print(f" - Median: {adata.obs['n_counts'].median():.1f}")
    print(f" - Mean: {adata.obs['n_counts'].mean():.1f}")
    print(f" - Std: {adata.obs['n_counts'].std():.1f}")
    print(f" - Min: {adata.obs['n_counts'].min():.1f}")
    print(f" - Max: {adata.obs['n_counts'].max():.1f}")
    print(f"\n Genes per cell:")
    print(f" - Median: {adata.obs['n_genes'].median():.1f}")
    print(f" - Mean: {adata.obs['n_genes'].mean():.1f}")
    
    return adata



def plot_qc_metrics(adata: sc.AnnData, save_path: str = None):
    """
    可视化质控指标分布
    """
    # --- 第一层缩进：所有代码都在 def 内部 ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    # 1. n_counts 分布 (左上角 [0,0])
    sns.histplot(adata.obs['n_counts'], bins=100, ax=axes[0, 0], color='steelblue')
    axes[0, 0].axvline(x=10, color='red', linestyle='--', linewidth=2, label='Min threshold (10)')
    axes[0, 0].set_xlabel('Total counts per cell')
    axes[0, 0].set_ylabel('Cell count')
    axes[0, 0].set_title('Distribution of Total Counts per Cell')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    # 2. n_genes 分布 (右上角 [0,1])
    sns.histplot(adata.obs['n_genes'], bins=100, ax=axes[0, 1], color='darkorange')
    axes[0, 1].set_xlabel('Number of genes detected')
    axes[0, 1].set_ylabel('Cell count')
    axes[0, 1].set_title('Distribution of Genes per Cell')
    axes[0, 1].grid(alpha=0.3)
    
    # 3. n_counts vs n_genes 散点图 (左下角 [1,0])
    axes[1, 0].scatter(
        adata.obs['n_counts'], 
        adata.obs['n_genes'],
        alpha=0.3, s=10, c='purple'
    )
    axes[1, 0].set_xlabel('Total counts')
    axes[1, 0].set_ylabel('Number of genes')
    axes[1, 0].set_title('Counts vs Genes per Cell')
    axes[1, 0].grid(alpha=0.3)
    
    # 4. 空间分布 (右下角 [1,1])
    if 'spatial' in adata.obsm:
        # --- 第二层缩进：在 if 内部 ---
        sc.pl.embedding(
            adata,
            basis='spatial',
            color='n_counts',
            ax=axes[1, 1],
            show=False,
            title='Spatial Distribution of Total Counts',
            cmap='viridis'
        )
    
    # 布局优化和保存 (回到第一层缩进，不论有没有空间数据都执行)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ QC plots saved to {save_path}")
    
    plt.show() # 确保在函数内部，否则调用函数时不会显示图



def preprocess_xenium(adata: sc.AnnData,
                      target_sum: int = 100,
                      use_log: bool = True,
                      use_scale: bool = True,
                      n_pcs: int = None) -> sc.AnnData:
    """
    Xenium 数据标准化预处理流程
    Based on Nature Methods 2025 best-practice workflow
    """
    
    print("\n" + "=" * 60)
    print("Standardized Preprocessing (Nature Methods 2025 Workflow)")
    print("=" * 60)
    
    # 步骤 1: Library size normalization
    print(f"\n[Step 1] Library size normalization (target_sum={target_sum})...")
    sc.pp.normalize_total(adata, target_sum=target_sum)
    print(" ✓ Completed")
    
    # 步骤 2: Log transformation
    if use_log:
        # if 内部的代码需要缩进
        print(f"\n[Step 2] Log transformation (log1p)...")
        sc.pp.log1p(adata)
        print(" ✓ Completed")
    else:
        # else 内部的代码需要缩进
        print(f"\n[Step 2] Log transformation skipped")
    
    # 步骤 3: Scaling
    if use_scale:
        print(f"\n[Step 3] Scaling (zero mean, unit variance)...")
        # Xenium 数据通常不建议在 scale 后做过强的剪切 (max_value=None)
        sc.pp.scale(adata, zero_center=True, max_value=None)
        print(" ✓ Completed")
    else:
        print(f"\n[Step 3] Scaling skipped")
    
    # 步骤 4: PCA
    print(f"\n[Step 4] PCA...")
    if n_pcs is None:
        # 逻辑：自动计算 PC 数，但上限设为 50
        n_pcs = min(adata.n_vars - 1, adata.n_obs - 1, 50)
        print(f" Using {n_pcs} components (min of n_vars-1, n_obs-1, 50)")
    else:
        print(f" Using {n_pcs} components")
    
    sc.tl.pca(adata, n_comps=n_pcs, random_state=42) # 注意这里用 tl.pca 
    print(f" ✓ PCA completed")
    # 打印前 5 个 PC 的解释变异率
    print(f" Explained variance ratio (PC1-5): {adata.uns['pca']['variance_ratio'][:5].round(3)}")
    
    return adata


def build_knn_and_cluster(adata: sc.AnnData,
                         n_neighbors: int = 16,
                         resolutions: list = [0.1,0.3,0.6],
                         use_rep: str = 'X_pca'
                         ) -> sc.AnnData:
    """
    构建 k-NN 图并进行 Louvain 聚类
    Based on Nature Methods 2025 parameters
    """
    
    print("\n" + "=" * 60)
    print("k-NN Graph Construction & Louvain Clustering")
    print("=" * 60)
    
    # --- 步骤 5: 构建 k-NN 图 ---
    print(f"\n[Step 5] Building k-NN graph (n_neighbors={n_neighbors})...")
    
    # 安全获取 PCA 维度，如果找不到则让 neighbors 自动决定
    try:
        n_pcs = adata.uns['pca']['params']['n_comps']
    except (KeyError, TypeError):
        n_pcs = None
        print(" ! PCA params not found, using default n_pcs")
    
    sc.pp.neighbors(
        adata,
        n_neighbors=n_neighbors,
        n_pcs=n_pcs,
        use_rep=use_rep,
        random_state=42
    )
    print(" ✓ k-NN graph constructed")
    
    # --- 步骤 6: Louvain 聚类 ---
    print(f"\n[Step 6] Leiden clustering (resolution={resolutions})...")
    
    for res in resolutions:
        key_added = f'leiden_res_{res}'
        sc.tl.leiden(
            adata, 
            resolution=res, 
            key_added=key_added, 
            random_state=42
        )
    
        n_clusters = len(adata.obs[key_added].unique())
    print(f" ✓ Clustering completed")
    
    return adata



###%pip install pybanksy

###%pip install pybanksy
import time
import sys
import os
import gc
import pandas as pd
from contextlib import contextmanager

@contextmanager
def silence_stdout():
    """用于屏蔽内部函数无用打印的上下文管理器"""
    new_target = open(os.devnull, "w")
    old_target = sys.stdout
    sys.stdout = new_target
    try:
        yield
    finally:
        sys.stdout = old_target
        new_target.close()

def RunSpatialcluster_Banksy(adata, spatial_key='spatial', lambda_val=0.2, res_list=[0.1, 0.2, 0.3], pca_dim=20, max_m=1):
    start_time = time.time()
    print(f"🚀 开始 BANKSY 空间聚类流水线 | 样本量: {adata.n_obs} | 目标分辨率: {res_list}")
    from banksy.main import median_dist_to_nearest_neighbour
    from banksy.initialize_banksy import initialize_banksy
    from banksy.embed_banksy import generate_banksy_matrix
    from banksy.main import concatenate_all
    from banksy_utils.umap_pca import pca_umap
    from banksy.cluster_methods import run_Leiden_partition
    #from banksy.plot_banksy import plot_results

    # 1. 环境准备
    if 'highly_variable' not in adata.var.columns:
        ad_run = adata.copy()
    else:
        ad_run = adata[:, adata.var.highly_variable].copy()

    ad_run.obs['x'] = ad_run.obsm[spatial_key][:, 0]
    ad_run.obs['y'] = ad_run.obsm[spatial_key][:, 1]
    coord_keys = ('x', 'y', spatial_key)

    # 2. 运行 BANKSY 核心步骤 (使用 silence_stdout 屏蔽其内部打印)
    with silence_stdout():
        # 初始化
        banksy_dict = initialize_banksy(
            ad_run, coord_keys=coord_keys, num_neighbours=15, max_m=max_m,
            nbr_weight_decay='scaled_gaussian', plt_edge_hist=False, 
            plt_nbr_weights=False, plt_agf_angles=False, plt_theta=False
        )

        # 生成矩阵
        banksy_dict, _ = generate_banksy_matrix(ad_run, banksy_dict, [lambda_val], max_m=max_m)

        # 降维
        pca_umap(banksy_dict, pca_dims=[pca_dim], add_umap=True, plt_remaining_var=False)

    print(f"📊 特征空间构建完成 (λ={lambda_val}, PCA={pca_dim})")

    # 3. 聚类迭代
    for res in res_list:
        loop_start = time.time()
        
        with silence_stdout():
            res_df, _ = run_Leiden_partition(banksy_dict, [res], num_nn=50, partition_seed=12345)
        
        # 提取标签
        labels_obj = res_df.loc[res_df.index[0], 'labels']
        col_name = f'banksy_res_{res}'
        
        adata.obs[col_name] = pd.Series(
            labels_obj.dense.flatten(), 
            index=ad_run.obs_names
        ).astype(str).astype('category')
        
        loop_end = time.time()
        print(f" ✅ 分辨率 {res} 聚类完成 | 耗时: {loop_end - loop_start:.2f}s | 聚类数: {len(adata.obs[col_name].unique())}")
        
        gc.collect()

    end_time = time.time()
    total_duration = end_time - start_time
    print(f"✨ 所有任务已完成！总运行时间: {total_duration/60:.2f} 分钟")
    
    return adata

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.stats import chi2_contingency
from statsmodels.stats.multitest import multipletests

def run_spatial_enrichment_pipeline(adata, center_cell, radius=50, cluster_key='Cell_type', 
                                    dot_scale=1000, cmap='Reds', save_path=None):
    """
    Xenium 空间邻域富集分析一站式函数
    """
    # 1. 空间检索与分组
    coords = adata.obsm['spatial']
    tree = cKDTree(coords)
    center_coords = coords[adata.obs[cluster_key] == center_cell]
    
    neighbor_indices = tree.query_ball_point(center_coords, radius)
    proximal_indices = np.unique([idx for sublist in neighbor_indices for idx in sublist])
    
    # 在副本上工作，保护原始 adata
    ad_work = adata.copy()
    prox_label = f'Proximal (<{radius}um)'
    dist_label = f'Distal (>{radius}um)'
    ad_work.obs['spatial_group'] = dist_label
    ad_work.obs.iloc[proximal_indices, ad_work.obs.columns.get_loc('spatial_group')] = prox_label
    
    # 排除中心细胞本身
    ad_no_center = ad_work[ad_work.obs[cluster_key] != center_cell].copy()
    
    # 2. 频数与比例统计
    freq_df = ad_no_center.obs.groupby(['spatial_group', cluster_key]).size().unstack(fill_value=0)
    freq_prop = freq_df.div(freq_df.sum(axis=1), axis=0).T
    
    # 3. 统计检验 (Chi-square)
    stats_list = []
    for cell_type in freq_df.columns:
        a = freq_df.loc[prox_label, cell_type]
        b = freq_df.loc[dist_label, cell_type]
        c = freq_df[cell_type].sum() - (a + b) # 这一列其他细胞在两组的总和
        # 更严谨的 2x2: [本细胞在P/D, 其他细胞在P/D]
        total_p = freq_df.loc[prox_label].sum()
        total_d = freq_df.loc[dist_label].sum()
        contingency = [[a, b], [total_p - a, total_d - b]]
        
        _, p, _, _ = chi2_contingency(contingency)
        
        # 计算 Fold Change
        p_prop = a / total_p if total_p > 0 else 0
        d_prop = b / total_d if total_d > 0 else 1e-5
        fc = p_prop / d_prop
        
        stats_list.append({'Cell_type': cell_type, 'P_value': p, 'Fold_Change': fc, 
                           'Prox_Prop': p_prop, 'Dist_Prop': d_prop})

    res = pd.DataFrame(stats_list).set_index('Cell_type')
    res['FDR'] = multipletests(res['P_value'], method='fdr_bh')[1]
    res['log2FC'] = np.log2(res['Fold_Change'].replace(0, 0.01))
    res['minus_log10P'] = -np.log10(res['FDR'] + 1e-20)
    
    # 4. 可视化绘图
    res = res.sort_values('log2FC') # 按差异倍数排序
    
    plt.figure(figsize=(7, 4))
    # 只有显著且富集的用深色描边
    scatter = plt.scatter(
        x=res['log2FC'], y=res.index, 
        s=res['Prox_Prop'] * dot_scale, 
        c=res['minus_log10P'], cmap=cmap, 
        alpha=0.8, edgecolors='black', linewidth=0.8
    )
    
    # 辅助线和样式
    plt.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.3)
    plt.xlabel('log2 (Fold Change: Proximal / Distal)', fontweight='bold', fontsize=12)
    plt.ylabel('Neighbor Cell Type', fontweight='bold', fontsize=12)
    plt.title(f'Niche Composition: Proximal to {center_cell} ({radius}μm)', fontweight='bold', fontsize=14, pad=20)
    
    # 颜色条
    cbar = plt.colorbar(scatter,shrink=0.4, aspect=30, pad=0.08)
    cbar.set_label('-log10 (FDR)', fontweight='bold')
    
    # 大小图例
    for p in [ 0.1, 0.2,0.3]:
        plt.scatter([], [], s=p*dot_scale, c='grey', alpha=0.3, label=f'{p*100:.0f}%', edgecolors='black')
    plt.legend(title="Proportion", loc='center left', bbox_to_anchor=(1.25, 0.5), frameon=False)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return res

# --- 调用演示 ---
# final_table = run_spatial_enrichment_pipeline(adata, center_cell='Treg', radius=50)
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_niche_composition_donuts(adata, cluster_key='cluster_cellcharter', cell_type_key='Cell_type', 
                                  n_cols=2, threshold=0.02, figsize_unit=5):
    """
    为每个空间域绘制细胞组成环形图 (Donut Chart)
    threshold: 比例低于此值的细胞类型将被归类为 'Others'
    """
    # 1. 计算每个 Cluster 内的细胞组成频数
    composition = adata.obs.groupby([cluster_key, cell_type_key]).size().unstack(fill_value=0)
    
    # 归一化为比例
    composition_prop = composition.div(composition.sum(axis=1), axis=0)
    
    clusters = composition_prop.index
    n_rows = int(np.ceil(len(clusters) / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * figsize_unit, n_rows * figsize_unit))
    axes = axes.flatten()
    
    # 获取全局颜色映射，保证同一个细胞类型在不同图中颜色一致
    unique_cells = adata.obs[cell_type_key].unique()
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_cells)))
    color_dict = dict(zip(unique_cells, colors))
    color_dict['Others'] = '#d3d3d3' # 灰色给 Others

    for i, cluster in enumerate(clusters):
        ax = axes[i]
        data = composition_prop.loc[cluster].sort_values(ascending=False)
        # 过滤低比例细胞，合并为 Others
        main_data = data[data >= threshold]
        others_val = data[data < threshold].sum()
        if others_val > 0:
            main_data['Others'] = others_val 
        # 准备绘图数据
        labels = [f"{l} ({v*100:.1f}%)" if v > 0.05 else "" for l, v in main_data.items()] # 只在图中显示大比例的文字
        plot_colors = [color_dict.get(l, '#d3d3d3') for l in main_data.index]
        
        # 绘制饼图
        wedges, texts = ax.pie(main_data, labels=labels, colors=plot_colors, 
                                startangle=140, pctdistance=0.85,
                                wedgeprops=dict(width=0.3, edgecolor='w')) # width=0.3 变成环形图
        
        ax.set_title(f"Cluster {cluster}\n(n={composition.sum(axis=1)[cluster]})", 
                     fontweight='bold', pad=10)

    # 隐藏多余的子图轴
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
        
    plt.tight_layout()
    plt.show()

# --- 调用 ---
# plot_niche_composition_donuts(adata, cluster_key='cluster_cellcharter', threshold=0.03)


"""
空间数据可视化工具
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, List, Dict


def plot_spatial_overview(
    adata,
    cluster_key: str = 'Cell_type',
    highlight_types: Optional[List[str]] = None,
    grid_interval: int = 1000,
    point_size: float = 0.1,
    figsize: tuple = (10, 8),
    title: Optional[str] = None,
    equal_aspect: bool = True,
) -> Dict[str, tuple]:
    """
    空间分布总览图，用于快速了解各细胞类型的空间分布范围，
    并通过坐标刻度辅助选择 ROI 区域。

    参数:
        adata: AnnData 对象，需包含 obsm['spatial'] 空间坐标
        cluster_key: adata.obs 中存储细胞类型的列名
        highlight_types: 需要突出显示的细胞类型列表，为 None 时自动选择全部
        grid_interval: 坐标刻度间隔（微米），默认 1000 μm
        point_size: 背景细胞点大小，默认 0.1
        figsize: 画布尺寸 (宽, 高)
        title: 图表标题，为 None 时使用默认标题
        equal_aspect: 是否保持坐标轴等比例

    返回:
        Dict[str, tuple]，键为细胞类型，值为 (x_min, x_max, y_min, y_max) 坐标范围
    """
    coords = adata.obsm['spatial']
    cell_types = adata.obs[cluster_key].unique()

    # 默认高亮全部类型
    if highlight_types is None:
        highlight_types = list(cell_types)

    # 构建颜色映射
    all_types = list(cell_types)
    n_types = len(all_types)
    cmap = plt.cm.tab20 if n_types <= 20 else plt.cm.tab20
    color_map = {t: cmap(i / n_types) for i, t in enumerate(all_types)}

    # 绘制画布
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor('white')

    # 1. 所有细胞作为浅灰背景
    ax.scatter(
        coords[:, 0], coords[:, 1],
        s=point_size, c='#DDDDDD', alpha=0.4,
        zorder=0,
    )

    # 2. 按类型依次绘制，每种类型独立颜色
    for i, ctype in enumerate(all_types):
        mask = adata.obs[cluster_key] == ctype
        color = color_map[ctype]
        # 高亮类型用更饱和的颜色
        if ctype in highlight_types:
            color = plt.cm.Set1(i / len(highlight_types))
            zorder = 3
            alpha = 0.7
            marker_size = point_size * 3
        else:
            zorder = 1
            alpha = 0.3
            marker_size = point_size

        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            s=marker_size, c=[color], alpha=alpha,
            zorder=zorder,
            label=ctype,
        )

    # 3. 坐标刻度（辅助读坐标）
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()

    ax.set_xticks(np.arange(x_min, x_max + grid_interval, grid_interval))
    ax.set_yticks(np.arange(y_min, y_max + grid_interval, grid_interval))
    ax.tick_params(axis='both', labelsize=7)
    ax.tick_params(axis='x', rotation=90)

    # 4. 边框简洁黑色
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(1)

    # 5. 比例尺
    if equal_aspect:
        ax.set_aspect('equal')

    # 6. 标签
    ax.set_xlabel('X (μm)', fontsize=10)
    ax.set_ylabel('Y (μm)', fontsize=10)
    ax.set_title(title or f'Spatial Overview (Grid: {grid_interval} μm)', fontweight='bold', fontsize=12)

    # 7. 图例放在右侧，避免遮挡
    ax.legend(
        loc='center left',
        bbox_to_anchor=(1.02, 0.5),
        fontsize=7,
        markerscale=3,
        frameon=False,
    )

    plt.tight_layout()
    plt.show()

    # 8. 返回每个细胞类型的坐标范围，方便后续提取 ROI
    range_info = {}
    for ctype in cell_types:
        mask = adata.obs[cluster_key] == ctype
        c_coords = coords[mask]
        if len(c_coords) > 0:
            range_info[ctype] = (
                c_coords[:, 0].min(), c_coords[:, 0].max(),
                c_coords[:, 1].min(), c_coords[:, 1].max(),
            )

    return range_info


def extract_roi(
    adata,
    x_range: tuple,
    y_range: tuple,
    cluster_key: str = 'Cell_type',
) -> 'AnnData':
    """
    根据坐标范围提取 ROI 区域的 AnnData。

    参数:
        adata: AnnData 对象
        x_range: (x_min, x_max) 微米坐标范围
        y_range: (y_min, y_max) 微米坐标范围
        cluster_key: adata.obs 中存储细胞类型的列名

    返回:
        裁剪后的 AnnData 子集
    """
    coords = adata.obsm['spatial']
    x_min, x_max = x_range
    y_min, y_max = y_range

    mask = (
        (coords[:, 0] >= x_min) & (coords[:, 0] <= x_max) &
        (coords[:, 1] >= y_min) & (coords[:, 1] <= y_max)
    )

    return adata[mask].copy()


















"""
空间距离 KDE 绘图模块

提供细胞群空间距离计算与密度分布绑图功能。
核心公式：D_k = min_j d(c_k, c_j)，即每个目标细胞到 Niche 细胞群的最近距离。
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from typing import List, Optional, Dict, Callable


# ============================================================================
# 距离计算
# ============================================================================

def calculate_shortest_distance(
    adata,
    target_subset_cell: str,
    niche_cells: List[str],
    cluster_key: str = 'Cell_type',
) -> np.ndarray:
    """
    计算目标细胞群中每个细胞到 Niche 细胞群的最短距离。

    公式：D_k = min_j d(c_k, c_j)

    参数:
        adata: AnnData 对象，需包含 obsm['spatial'] 空间坐标
        target_subset_cell: 目标细胞类型名称（adata.obs[cluster_key] 中的值）
        niche_cells: Niche 构成细胞群列表（如 ['Treg', 'T', 'B']）
        cluster_key: adata.obs 中存储细胞类型的列名

    返回:
        每个目标细胞的最近邻距离数组，长度 = 目标细胞数量

    异常:
        ValueError: 当指定细胞类型在数据中不存在时
    """
    coords = adata.obsm['spatial']

    # 校验坐标数据
    if coords is None or coords.shape[0] == 0:
        raise ValueError("adata.obsm['spatial'] 为空或不存在，请检查数据。")

    # 校验 cluster_key 列存在
    if cluster_key not in adata.obs.columns:
        raise ValueError(f"adata.obs 中不存在列 '{cluster_key}'。")

    # 1. 提取 Niche 细胞坐标
    niche_mask = adata.obs[cluster_key].isin(niche_cells)
    niche_coords = coords[niche_mask]

    if len(niche_coords) == 0:
        raise ValueError(
            f"Niche 细胞群 {niche_cells} 在 {cluster_key} 中未找到任何细胞。"
        )

    # 2. 提取目标细胞坐标
    target_mask = adata.obs[cluster_key] == target_subset_cell
    target_coords = coords[target_mask]

    if len(target_coords) == 0:
        raise ValueError(
            f"目标细胞类型 '{target_subset_cell}' 在 {cluster_key} 中未找到任何细胞。"
        )

    # 3. KDTree 最近邻查询
    tree = cKDTree(niche_coords)
    distances, _ = tree.query(target_coords, k=1)

    return distances


# ============================================================================
# 统计量计算（与绘图解耦）
# ============================================================================

def compute_spatial_distance_stats(
    adata,
    center_types: List[str],
    target_types: List[str],
    cluster_key: str = 'Cell_type',
    distance_func: Optional[Callable] = None,
) -> pd.DataFrame:
    """
    计算目标细胞群到中心细胞群的最短距离统计量。

    参数:
        adata: AnnData 对象
        center_types: Niche 细胞群列表
        target_types: 目标细胞类型列表
        cluster_key: adata.obs 中存储细胞类型的列名
        distance_func: 距离计算函数，默认为 calculate_shortest_distance

    返回:
        DataFrame，列：Cell_Type, Mean, SD, Median, N_Cells
    """
    if distance_func is None:
        distance_func = calculate_shortest_distance

    records = []
    for t_type in target_types:
        dists = distance_func(adata, t_type, center_types, cluster_key)

        if len(dists) == 0:
            records.append({
                'Cell_Type': t_type,
                'Mean': np.nan,
                'SD': np.nan,
                'Median': np.nan,
                'N_Cells': 0,
            })
        else:
            records.append({
                'Cell_Type': t_type,
                'Mean': np.mean(dists),
                'SD': np.std(dists),
                'Median': np.median(dists),
                'N_Cells': len(dists),
            })

    return pd.DataFrame(records)


# ============================================================================
# 绑图函数
# ============================================================================

def plot_spatial_distance_kde(
    adata,
    center_types: List[str],
    target_types: List[str],
    cluster_key: str = 'Cell_type',
    max_dist: float = 200,
    palette: Optional[Dict[str, str]] = None,
    figsize: tuple = (8, 4),
    distance_func: Optional[Callable] = None,
    annotate_stats: bool = True,
    fill: bool = True,
) -> pd.DataFrame:
    """
    绘制目标细胞群到中心细胞群的最短距离密度分布图。

    参数:
        adata: AnnData 对象
        center_types: Niche 细胞群，如 ['Carcinoma'] 或 ['Treg', 'T']
        target_types: 要观察分布的目标细胞类型，如 ['iCAF', 'myCAF']
        cluster_key: adata.obs 中存储细胞类型的列名
        max_dist: 横轴最大距离（超出部分被截断）
        palette: 颜色字典，为 None 时使用 seaborn 默认调色板
        figsize: 每个分面的尺寸 (宽, 高)
        distance_func: 距离计算函数，默认为 calculate_shortest_distance
        annotate_stats: 是否在子图上标注均值 ± 标准差
        fill: 是否填充 KDE 曲线下方区域

    返回:
        统计量 DataFrame，列：Cell_Type, Mean, SD, Median, N_Cells
    """
    if distance_func is None:
        distance_func = calculate_shortest_distance

    # ------------------------------------------------------------------
    # 1. 一次遍历：同时收集绘图数据和统计记录
    # ------------------------------------------------------------------
    plot_records: List[Dict] = []
    stats_records: List[Dict] = []

    for t_type in target_types:
        dists = distance_func(adata, t_type, center_types, cluster_key)

        if len(dists) == 0:
            stats_records.append({
                'Cell_Type': t_type,
                'Mean': np.nan,
                'SD': np.nan,
                'Median': np.nan,
                'N_Cells': 0,
            })
            continue

        plot_records.append({'dist': dists, 'type': t_type})
        stats_records.append({
            'Cell_Type': t_type,
            'Mean': np.mean(dists),
            'SD': np.std(dists),
            'Median': np.median(dists),
            'N_Cells': len(dists),
        })

    stats_df = pd.DataFrame(stats_records)

    if not plot_records:
        raise ValueError(
            f"未在 {cluster_key} 中找到任何指定细胞类型的数据。"
            f"center_types={center_types}, target_types={target_types}"
        )

    # 拼接绘图数据（避免循环中反复 concat）
    plot_df = pd.concat(
        [pd.DataFrame({'dist': r['dist'], 'type': r['type']}) for r in plot_records],
        ignore_index=True,
    )

    # ------------------------------------------------------------------
    # 2. 配色与绑图
    # ------------------------------------------------------------------
    sns.set_style("whitegrid")

    # 若未提供 palette，为每个 target_type 分配独立颜色
    if palette is None:
        n_types = len(target_types)
        color_palette = sns.color_palette("husl", n_types)
        palette = dict(zip(target_types, color_palette))

    g = sns.FacetGrid(
        plot_df,
        row="type",
        hue="type",
        aspect=2,
        height=figsize[1],
        palette=palette,
    )

    kde_kws = {"clip": (0, max_dist)}
    if fill:
        g.map(sns.kdeplot, "dist", fill=True, alpha=0.6, lw=2, **kde_kws)
    else:
        g.map(sns.kdeplot, "dist", lw=2, **kde_kws)

    # 去掉每行 "type = xxx" 的冗余前缀标签
    g.set_titles("")
    g.figure.subplots_adjust(hspace=0.4)

    # ------------------------------------------------------------------
    # 3. 遍历子图：标注统计量和样式调整
    # ------------------------------------------------------------------
    for idx, ax in enumerate(g.axes.flat):
        if idx >= len(target_types):
            break

        curr_type = target_types[idx]
        data = plot_df[plot_df['type'] == curr_type]['dist']
        mean_val = data.mean()
        std_val = data.std()

        ax.set_xlim(-max_dist * 0.02, max_dist)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_xlabel('')
        ax.tick_params(left=False)  # 去掉Y轴刻度线，更干净

        # 简洁黑色方框
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(1)

        # 标题放正上方
        ax.set_title(
            curr_type,
            loc='center',
            fontsize=12,
            fontweight='bold',
            pad=12,
        )

        # 均值虚线
        ax.axvline(mean_val, color='black', linestyle='--', linewidth=1.5)

        # 统计文本：放右上角
        if annotate_stats:
            ax.text(
                0.97, 0.92,
                f"{mean_val:.2f} ± {std_val:.2f} μm",
                transform=ax.transAxes,
                ha='right',
                va='top',
                fontsize=11,
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7),
            )

    # ------------------------------------------------------------------
    # 4. 统一标签与布局
    # ------------------------------------------------------------------
    center_label = ', '.join(center_types)
    g.set_axis_labels(f"Distance to {center_label} (μm)", fontsize=11)
    g.figure.align_labels()  # 标签对齐

    return stats_df



"""
CellCharter 空间聚类流水线

需在 cellcharter 环境（包含 scvi-tools、squidpy、cellcharter）中运行。
"""

import time
import gc
import sys
import os
import random
from contextlib import contextmanager
from typing import Optional

import numpy as np
import pandas as pd


@contextmanager
def _silence_stdout():
    """屏蔽底层库冗余输出的上下文管理器"""
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


def _print_progress(msg: str, flush: bool = True):
    """带时间戳的进度打印，兼容无交互环境。"""
    prefix = time.strftime("%H:%M:%S")
    print(f"[{prefix}] {msg}", flush=flush)


def run_cellcharter_pipeline(
    adata,
    batch_key: str = 'patient',
    layer: str = 'counts',
    n_layers: int = 3,
    min_k: int = 2,
    max_k: int = 10,
    max_runs: int = 10,
    out_key: str = 'cluster_cellcharter',
    random_seed: int = 12345,
) -> 'sc.AnnData':
    """
    CellCharter 自动 K 值空间聚类流水线。

    流程：scVI 批次校正 → Delaunay 空间邻居图 → 邻域特征聚合 → AutoK 稳定性搜索 → 聚类。

    参数:
        adata: AnnData 对象，需包含 obsm['spatial'] 空间坐标和 obs[batch_key] 批次标签
        batch_key: 批次标签列名（用于 scVI 校正和空间邻居构建）
        layer: 用于 scVI 的表达层（默认 'counts'）
        n_layers: 邻域特征聚合层数
        min_k: 自动 K 搜索最小簇数
        max_k: 自动 K 搜索最大簇数
        max_runs: AutoK 稳定性搜索次数
        out_key: 聚类结果写入 adata.obs 的列名
        random_seed: 随机种子，保证结果可重复

    返回:
        输入 adata（inplace 更新 obs[out_key] 和 obsm['X_scVI']、obsm['X_cellcharter']）

    依赖:
        需在包含 scvi-tools、squidpy、cellcharter 的 conda 环境运行：
        conda activate cellcharter
    """
    # ── 种子设定 ──────────────────────────────────────────────────────────────
    try:
        from lightning.pytorch import seed_everything as _seed
        _seed(random_seed)
    except ImportError:
        try:
            from pytorch_lightning import seed_everything as _seed
            _seed(random_seed)
        except ImportError:
            random.seed(random_seed)
            np.random.seed(random_seed)
            _print_progress(
                f"⚠️ 未检测到 lightning，种子已通过 random/numpy 设定"
            )

    # ── 前置校验 ──────────────────────────────────────────────────────────────
    if layer not in adata.layers:
        raise ValueError(
            f"layer='{layer}' 在 adata.layers 中不存在。"
            f"可用: {list(adata.layers.keys())}"
        )

    if batch_key not in adata.obs.columns:
        raise ValueError(
            f"batch_key='{batch_key}' 在 adata.obs 中不存在。"
            f"可用列: {list(adata.obs.columns)}"
        )

    if 'spatial' not in adata.obsm:
        raise ValueError("adata.obsm['spatial'] 不存在，请检查数据。")

    n_batches = adata.obs[batch_key].nunique()
    _print_progress(
        f"🚀 CellCharter 启动 | 细胞: {adata.n_obs:,} | 批次: {n_batches} | K: {min_k}–{max_k}"
    )

    t_start = time.time()

    # ── 动态导入（避免顶层依赖缺失时报错）────────────────────────────────────
    import scanpy as sc
    import scvi
    import squidpy as sq
    import cellcharter as cc

    scvi.settings.seed = random_seed

    # ── Step 1: scVI 批次校正 ────────────────────────────────────────────────
    if 'X_scVI' not in adata.obsm:
        t0 = time.time()
        _print_progress("── scVI 潜在表示提取中...", flush=False)

        try:
            with _silence_stdout():
                scvi.model.SCVI.setup_anndata(
                    adata,
                    layer=layer,
                    batch_key=batch_key,
                )
                model = scvi.model.SCVI(adata)
                model.train(
                    early_stopping=True,
                    enable_progress_bar=False,
                )
                adata.obsm['X_scVI'] = (
                    model.get_latent_representation().astype(np.float32)
                )
            del model
            gc.collect()

            _print_progress(
                f" ✅ scVI 完成 ({time.time() - t0:.1f}s) | shape: {adata.obsm['X_scVI'].shape}"
            )
        except Exception as e:
            raise RuntimeError(
                f"scVI 批次校正失败: {e}"
            ) from e
    else:
        _print_progress("── 跳过 scVI（已存在 X_scVI）")

    # ── Step 2: Delaunay 空间邻居图 ──────────────────────────────────────────
    t0 = time.time()
    _print_progress("── 构建 Delaunay 空间邻居图...", flush=False)

    try:
        sq.gr.spatial_neighbors(
            adata,
            library_key=batch_key,
            coord_type='generic',
            delaunay=True,
            spatial_key='spatial',
            percentile=99,
        )
        _print_progress(f" ✅ 邻居图完成 ({time.time() - t0:.1f}s)")
    except Exception as e:
        raise RuntimeError(f"空间邻居图构建失败: {e}") from e

    # ── Step 3: 邻域特征聚合 ─────────────────────────────────────────────────
    t0 = time.time()
    _print_progress(f"── 聚合邻域特征 (layers={n_layers})...", flush=False)

    try:
        cc.gr.aggregate_neighbors(
            adata,
            n_layers=n_layers,
            use_rep='X_scVI',
            out_key='X_cellcharter',
            sample_key=batch_key,
        )
        _print_progress(
            f" ✅ 特征聚合完成 ({time.time() - t0:.1f}s) | "
            f"shape: {adata.obsm['X_cellcharter'].shape}"
        )
    except Exception as e:
        raise RuntimeError(f"邻域特征聚合失败: {e}") from e

    # ── Step 4: AutoK 稳定性搜索 ─────────────────────────────────────────────
    t0 = time.time()
    _print_progress(
        f"── Stability-based AutoK 搜索 (K={min_k}–{max_k}, runs={max_runs})...",
        flush=False,
    )

    try:
        with _silence_stdout():
            autok = cc.tl.ClusterAutoK(
                n_clusters=(min_k, max_k),
                max_runs=max_runs,
                convergence_tol=0.001,
            )
            autok.fit(adata, use_rep='X_cellcharter')
    except Exception as e:
        raise RuntimeError(f"AutoK 搜索失败: {e}") from e

    # 写入聚类结果
    adata.obs[out_key] = autok.predict(adata, use_rep='X_cellcharter')
    best_k = adata.obs[out_key].nunique()

    _print_progress(
        f" ✅ AutoK 完成 | 最佳 K={best_k} ({time.time() - t0:.1f}s)"
    )

    # ── 完成 ────────────────────────────────────────────────────────────────
    total_min = (time.time() - t_start) / 60
    _print_progress(
        f"✨ 完成！总耗时: {total_min:.2f} min | "
        f"聚类标签已写入 adata.obs['{out_key}']"
    )

    return adata



"""
空间配受体（Ligand-Receptor）分析流水线

基于 squidpy 的空间约束置换检验，返回原生结果字典，
配合 sq.pl.ligrec() 做灵活可视化。

可视化示例:
    >>> res = run_spatial_lr_pipeline(adata, cluster_key='cluster_cellcharter')
    >>> sq.pl.ligrec(res,
    ...     cluster_key='cluster_cellcharter',
    ...     source_groups=['Treg'],
    ...     target_groups=['Carcinoma'],
    ...     means_range=(0.5, np.inf),
    ...     pvalue_threshold=0.05,
    ...     annotate=True,
    ...     dpi=300)
"""

import time
from typing import Optional

import numpy as np


def run_spatial_lr_pipeline(
    adata,
    cluster_key: str = 'cluster_cellcharter',
    sample_key: str = 'patient',
    n_perms: int = 1000,
    n_jobs: int = 8,
    seed: int = 12345,
    interactions: Optional[str] = None,
) -> dict:
    """
    空间配受体（L/R）置换检验。

    在每个空间域内对所有细胞类型两两做配受体置换检验，
    识别在空间上显著共定位的 L/R 对。

    参数:
        adata: AnnData 对象，需包含 obs[cluster_key] 和 obsm['spatial']
        cluster_key: 空间域 / 细胞类型列名
        sample_key: 样本批次列名（用于构建空间邻居图）
        n_perms: 置换检验次数（建议 500-1000）
        n_jobs: 并行核数（-1 = 全部可用 CPU）
        seed: 随机种子，保证可重复
        interactions: L/R 数据库来源，None 则使用 OmniPath 最新版

    返回:
        dict: sq.gr.ligrec() 原生结果，可直接传入 sq.pl.ligrec() 做可视化。
              包含键：'means'（平均表达）、'pvalues'（原始 p 值）等。

    依赖:
        conda install squidpy
        或 pip install squidpy

    示例:
        res = run_spatial_lr_pipeline(adata, cluster_key='cluster_cellcharter')
        # 全部配受体热图
        sq.pl.ligrec(res, cluster_key='cluster_cellcharter', pvalue_threshold=0.05)
        # 只看特定细胞互作
        sq.pl.ligrec(res, cluster_key='cluster_cellcharter',
                     source_groups=['Treg'], target_groups=['Carcinoma'],
                     means_range=(0.5, np.inf), pvalue_threshold=0.05)
    """
    import squidpy as sq

    t_start = time.time()

    # ── 前置校验 ──────────────────────────────────────────────────────────────
    if cluster_key not in adata.obs.columns:
        raise ValueError(
            f"cluster_key='{cluster_key}' 不在 adata.obs 中。"
            f"可用列: {list(adata.obs.columns)}"
        )

    if 'spatial' not in adata.obsm:
        raise ValueError("adata.obsm['spatial'] 不存在，请检查数据。")

    n_clusters = adata.obs[cluster_key].nunique()

    sample_id = (
        str(adata.obs[sample_key].unique()[0])
        if sample_key in adata.obs
        else "Sample"
    )

    _print(
        f"📡 LR 分析启动 | 样本: {sample_id} | "
        f"细胞: {adata.n_obs:,} | 空间域: {n_clusters}"
    )

    # ── Step 1: 空间邻居图 ───────────────────────────────────────────────────
    if 'spatial_neighbors' not in adata.uns:
        t0 = time.time()
        _print("── 构建 Delaunay 空间邻居图...", flush=False)

        sq.gr.spatial_neighbors(
            adata,
            library_key=sample_key if sample_key in adata.obs else None,
            coord_type='generic',
            delaunay=True,
            spatial_key='spatial',
            percentile=99,
        )
        _print(f" ✅ 邻居图完成 ({time.time() - t0:.1f}s)")
    else:
        _print("── 跳过邻居图（已存在）")

    # ── Step 2: L/R 置换检验 ─────────────────────────────────────────────────
    t0 = time.time()
    _print(
        f"── 置换检验进行中 (n_perms={n_perms}, n_jobs={n_jobs})...",
        flush=False,
    )

    res = sq.gr.ligrec(
        adata,
        cluster_key=cluster_key,
        use_raw=False,
        interactions=interactions,
        n_perms=n_perms,
        n_jobs=n_jobs,
        seed=seed,
        copy=True,
        corr_method=None,
        threshold=0.01,
        complex_policy='min',
    )

    _print(f" ✅ 计算完成 ({time.time() - t0:.1f}s)")

    if res is None or 'means' not in res:
        raise RuntimeError(
            "sq.gr.ligrec 返回结果为空。"
            "请检查 cluster_key 是否有效，或尝试安装最新版 squidpy。"
        )

    # ── 完成 ─────────────────────────────────────────────────────────────────
    total_min = (time.time() - t_start) / 60
    n_pairs = res['means'].shape[0]
    _print(
        f"✨ 完成！总耗时: {total_min:.2f} min | "
        f"已测试配受体对: {n_pairs}"
    )

    return res


def _print(msg: str, flush: bool = True):
    prefix = time.strftime("%H:%M:%S")
    print(f"[{prefix}] {msg}", flush=flush)

