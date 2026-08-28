# 系统架构

## 1. 目标

系统用于 wafer particle map 的历史案例检索，不直接声称自动诊断 root cause。输入一张新 map，输出 Top-K 历史相似案例、comment、metadata 和可解释分项相似度。

## 2. 模块

```text
GUI / CLI
   ↓
WaferFAService
   ├─ image_io.py        wafer 检测、归一化、红点分割
   ├─ features.py        空间特征
   ├─ similarity.py      分项相似度 + 加权融合
   └─ db.py              SQLite 案例与 feature blob
```

## 3. 为什么先不用深度学习

当前目标数据量只有几十张。这个量级直接训练 CNN/Siamese 极易过拟合，而且很难解释为什么两张 map 相似。v0.1.0 先建立可验证 baseline：几何归一化 + 空间统计特征 + 可解释评分。

## 4. 数据流

### 入库

1. 读取原图。
2. 自动检测 wafer 圆。
3. 归一化到 512×512。
4. HSV 分割红色 particle。
5. connected components 得到每个 particle 中心。
6. 转换到 wafer 归一化坐标 `[-1,1]²`。
7. 提取多路特征。
8. 原图副本、归一化图、comment、metadata、feature blob 写入 SQLite。

### 检索

1. 对 query 执行同样预处理。
2. 遍历历史库。
3. 计算 6 个分项相似度。
4. 加权求总分。
5. 降序返回 Top-K。

## 5. 性能边界

几十到几千张案例时线性遍历足够简单可靠。超过约 10k–50k 张后，再根据实测延迟引入 FAISS/HNSW，而不是提前增加系统复杂度。

## 6. 可扩展点

- 特征版本化：`FEATURE_VERSION`。
- 后续增加 learned embedding，不破坏现有数据库语义。
- metadata 可扩展 tool/chamber/recipe/process step 等字段。
- 后续增加筛选器：先按 metadata 过滤，再做相似度排名。
