# 验证计划

## 1. 目标

验证系统真正解决的是“历史相似案例检索”，而不是只在合成图片上看起来漂亮。

## 2. 最低验收集

建议真实现场至少收集：

- center ≥ 10
- line/directional ≥ 10
- arc/sector ≥ 10
- ring/edge ≥ 10
- random ≥ 10
- mixed/unknown ≥ 10

每张至少由工程师给出 comment；最好额外标注 1–3 张“最相似历史案例”作为人工 ground truth。

## 3. 指标

### Retrieval Recall@3

对每张 query，如果人工认定的相似案例至少一张出现在系统 Top 3，则命中。

```text
Recall@3 >= 85%  → 进入现场试用
70%–85%          → 调权重/阈值后复测
<70%              → 停止加 UI 功能，先修算法/数据
```

### Top-1 Agreement

工程师是否认为第一名是合理参考。

目标：`>= 70%`。

### Scale robustness

同一 map 人工缩放到 0.6× / 1.5× / 2.0× 后重新检索，原案例应稳定进入 Top 3。

目标：`>= 95%`。

### Latency

历史库 1,000 张时单次 Top-3 检索：

- 目标：< 500 ms（普通办公 CPU，SSD）
- 超过 1 s 才考虑向量索引优化

## 4. 失败条件

以下任一成立，不应声称 MVP 可用：

- wafer 圆检测在真实截图上成功率 < 98%。
- particle 分割漏检/误检严重，数量误差 > 15%。
- Recall@3 < 70%。
- 工程师发现相似度对 root-cause 无实际参考意义。

## 5. 48 小时验证

1. 找 30–60 张真实历史 map。
2. 每张附 comment。
3. 随机留出 20% 做 query。
4. 跑 Top-3。
5. 两名工程师盲评 Top-3 是否有参考价值。
6. 记录失败案例，优先分析是 circle、particle segmentation、feature 还是权重问题。
