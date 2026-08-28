# Wafer Particle Pattern Historical Case Retrieval / FA Assistant

一个面向晶圆 particle map 的**历史案例相似检索与 FA 辅助工具**。目标不是把 wafer map 生硬分类成 `line / center / random`，而是把新出现的 particle 空间分布与历史案例库逐一比较，返回最相似的 Top-K 案例、相似度分解和历史 comment，帮助工程师快速关联过去的 FA 结论、机台/腔体/recipe 与处理措施。

> 当前版本：`v0.1.0`，优先做可解释、少样本、离线可运行的传统空间特征检索基线。几十张历史图即可开始使用，不要求先训练深度学习模型。

## 核心流程

```text
新 particle map
    ↓
wafer 圆检测 + 几何归一化
    ↓
红色 particle 分割 + 粒子中心提取
    ↓
空间特征
  - radial histogram
  - angular histogram
  - density map
  - centroid / radial stats
  - PCA lineality
  - cluster metrics
    ↓
与 SQLite 历史库中的特征逐一比较
    ↓
加权相似度排序
    ↓
Top 3 历史图片 + comment + metadata + 分项得分
```

## 为什么不把 ORB 当主算法

ORB 擅长带有稳定局部纹理/角点的图像。particle map 里的大量红点局部外观高度相似，真正有信息的是**点的全局空间分布**，而不是每个点附近的局部描述子。因此本项目先把图像还原成归一化 wafer 坐标中的 particle 点集，再计算空间统计特征。

## 当前能力

- 自动检测 wafer 圆；失败时使用保守中心圆 fallback。
- 将任意分辨率截图映射到统一 wafer 坐标系，降低缩放差异影响。
- HSV 双红区间提取红色 particle，并自动过滤 wafer 外内容。
- 生成可解释的多路空间特征。
- SQLite 历史案例库：图片、comment、metadata、特征一起保存。
- Top-K 相似检索，返回总分和分项得分。
- Tkinter 桌面 GUI，适合离线 Windows 环境。
- CLI：初始化、添加案例、检索、重建索引、生成 demo 数据。
- 合成 center / line / ring / arc / random 数据，便于没有真实 fab 数据时先验收流程。
- pytest + GitHub Actions CI。

## 安装

推荐 Python 3.10–3.13。Windows：

```powershell
py -m venv .venv
.\.venv\Scripts\activate
python -m pip install -U pip
pip install -e .
```

## 30 秒跑 Demo

```powershell
python -m wafer_fa.cli init
python -m wafer_fa.cli demo --count-per-pattern 8
python app.py
```

GUI 中：

1. 点击“导入历史案例”，选择图片并填写 comment；或先运行 `demo`。
2. 点击“选择查询图像”。
3. 系统显示 Top 3 历史案例及相似度。
4. 结果中的百分比是**相似度评分，不是模型置信度/概率**。

## CLI

### 初始化数据库

```bash
python -m wafer_fa.cli init
```

### 添加历史案例

```bash
python -m wafer_fa.cli add path/to/map.png \
  --comment "Chamber 3 管路污染，wet clean 后恢复" \
  --meta tool=OX-01 --meta chamber=C3 --meta recipe=RCP-A
```

### 检索 Top 3

```bash
python -m wafer_fa.cli search path/to/query.png --top-k 3
```

### 重建全部特征

算法参数更新后：

```bash
python -m wafer_fa.cli rebuild
```

### 生成合成演示数据

```bash
python -m wafer_fa.cli demo --count-per-pattern 10
```

## 相似度

默认总分：

```text
0.35 * density
0.20 * radial
0.10 * angular
0.15 * summary
0.10 * cluster
0.10 * particle_count
```

每个分项均映射到 `[0, 1]`，最终乘 100 展示。权重集中定义在 `src/wafer_fa/config.py`，后续可以用真实工程师标注的“相似/不相似”案例校准，而不是拍脑袋长期固定。

## 数据目录

运行后默认生成：

```text
data/
  wafer_fa.db
  images/        # 入库案例副本
  normalized/    # 归一化预览
  demo/          # 合成 demo 图
```

原始图片不会被修改。

## 推荐的数据字段

最低要求只有 `image + comment`。建议逐步补充：

- `tool`
- `chamber`
- `recipe`
- `lot`
- `wafer_id`
- `process_step`
- `timestamp`
- `fa_root_cause`
- `action`
- `result`

这样 Top-K 检索才会逐渐从“找长得像的图”变成真正的历史 FA 知识库。

## 设计边界

### v0.1.0 做

- 单张 wafer particle map。
- 红色 particle / 明显 wafer 圆边界。
- 少样本历史检索。
- 缩放归一化。
- 离线 Windows 运行。

### v0.1.0 不做

- 不声称自动给出 root cause。
- 不把相似度当概率。
- 不用几十张数据硬训练 CNN。
- 不做 MES/EDA 生产系统直连。
- 不保证对任意配色的 vendor map 零配置工作。

## 下一阶段

当真实案例积累到数百到数千张后，再加：

1. Siamese / Triplet embedding；
2. 自监督 wafer representation learning；
3. FA 工程师 pairwise feedback 学习权重；
4. FAISS/Qdrant 向量索引；
5. tool/chamber/recipe 条件过滤；
6. 相似案例 + 处理结果闭环统计。

## 文档

- [系统架构](docs/ARCHITECTURE.md)
- [算法说明](docs/ALGORITHM.md)
- [数据模型](docs/DATA_SCHEMA.md)
- [验证计划](docs/VALIDATION_PLAN.md)

## License

MIT
