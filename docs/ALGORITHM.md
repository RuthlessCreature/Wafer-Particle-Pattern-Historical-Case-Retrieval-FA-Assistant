# 算法说明

## 1. 几何归一化

检测 wafer 圆心 `(cx, cy)` 和半径 `r`。particle 像素坐标 `(x, y)` 映射为：

```text
x' = (x - cx) / r
y' = (y - cy) / r
```

因此截图从 800px 缩放成 1600px，只要 wafer 圆仍可检测，particle 的归一化坐标基本不变。

## 2. Particle 分割

默认假设 particle 为红色，使用 HSV 两段 hue 范围覆盖红色环绕区间。wafer 圆外全部屏蔽，避免标题、按钮、注释文字被误识别。

真实 vendor 配色若不同，应调整 `FeatureConfig` 中 HSV 阈值，而不是重新训练模型。

## 3. 特征

### Radial histogram

`r_i = sqrt(x_i² + y_i²)`，默认 20 个环区。

适合区分：center / edge / ring / random。

### Angular histogram

`theta_i = atan2(y_i, x_i)`，默认 36 个扇区。

适合区分：单侧聚集、局部 arc、方向性污染。

### Density map

归一化 wafer 切成 16×16 网格，统计粒子密度并做轻微 3×3 平滑。它是当前权重最高的一路，因为保留了最多二维空间结构。

### Summary

包含：

- centroid x/y
- radial mean/std
- radial q25/q50/q75/q90
- PCA lineality
- PCA principal angle
- angular resultant concentration
- ringness

### Cluster

依赖轻量 DBSCAN-like 聚类，输出：

- cluster count
- largest cluster ratio
- clustered ratio
- noise ratio
- cluster center spread

## 4. 相似度

默认：

```text
S = 0.35 Sdensity
  + 0.20 Sradial
  + 0.10 Sangular
  + 0.15 Ssummary
  + 0.10 Scluster
  + 0.10 Scount
```

直方图/密度使用 cosine similarity。summary/cluster 使用 RMS 距离经指数函数映射到 `[0,1]`。particle count 使用对称 log-ratio。

## 5. 重要限制

- 当前角度是绝对坐标。如果 vendor 截图会任意旋转，应先确认 notch/flat 方向语义，再决定是否做旋转不变匹配。
- 相似度不是统计概率。
- 两张图长得像不代表 root cause 相同，所以系统输出历史 comment，而不是自动宣判原因。
