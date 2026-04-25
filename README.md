# Xenium Pipline

Xenium 空间转录组数据分析流水线，基于 Nature Methods 2025 标准工作流。

## 功能模块

| 模块 | 环境 | 说明 |
|------|------|------|
| 质控过滤 | ViaEnv | basic_qc_filtering |
| 预处理 | ViaEnv | preprocess_xenium (Library Norm → Log1p → Scale → PCA) |
| Leiden 聚类 | ViaEnv | build_knn_and_cluster |
| BANKSY 空间聚类 | ViaEnv | RunSpatialcluster_Banksy |
| 最短距离 KDE | ViaEnv | plot_spatial_distance_kde |
| 空间富集分析 | ViaEnv | run_spatial_enrichment_pipeline |
| 环形图 | ViaEnv | plot_niche_composition_donuts |
| CellCharter 自动 K | cellcharter | run_cellcharter_pipeline |
| 配受体分析 | cellcharter | run_spatial_lr_pipeline |

## 环境配置

### ViaEnv（上游分析）

```bash
conda activate ViaEnv
```

### cellcharter（下游分析）

```bash
conda activate cellcharter
```

> ⚠️ 两个环境依赖包冲突，必须在各自环境中运行对应步骤。

## 快速开始

### 1. ViaEnv：上游分析

```python
import xenium_pipline as xp

# 质控
adata = xp.basic_qc_filtering(adata, min_reads=10, min_genes=5)

# 预处理
adata = xp.preprocess_xenium(adata)

# Leiden 聚类
adata = xp.build_knn_and_cluster(adata, resolutions=[0.1, 0.3, 0.6])

# BANKSY 聚类
adata = xp.RunSpatialcluster_Banksy(adata, res_list=[0.1, 0.2, 0.6])

# 细胞注释
label_map = {'0': 'Cancer Cell', '1': 'mFib', '2': 'Treg'}
adata.obs['Cell_type'] = adata.obs['leiden_res_0.1'].map(label_map)
```

### 2. cellcharter：空间聚类（换环境重启 kernel）

```python
adata = xp.run_cellcharter_pipeline(adata, batch_key='patient', n_layers=3)
```

### 3. 配受体分析

```python
res = xp.run_spatial_lr_pipeline(adata, cluster_key='Cell_type', n_perms=1000, n_jobs=8)
```

## 项目结构

```
Xenium-pipline/
├── scripts/
│   └── xenium_pipline.py      # 所有分析函数
├── notebooks/
│   ├── Xenium pipline code.ipynb   # 完整流程 Notebook
│   └── run_cellcharter.ipynb       # CellCharter 独立运行
└── docs/
    └── 使用文档.md               # 完整使用说明
```

## 文档

详细说明见 [docs/使用文档.md](docs/使用文档.md)，包含：
- 两个环境的完整配置
- 所有函数参数说明
- 标准分析流程代码
- squidpy 可视化速查
- 常见问题解答

## 数据说明

`*.h5ad` 数据文件不在 GitHub 仓库中，请将数据放在 `data/` 目录下：

```
data/
└── P8.h5ad
```
