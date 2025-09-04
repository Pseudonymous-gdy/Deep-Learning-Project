太好了。我把你给的“冲刺计划”思路（快速复现→小而清晰的创新→密集实验→规范写作）跟我们前面讨论的 **Bayesian 路由 + 稀疏 MoE** 方案融合，做成**1 个明确选题 + 10 周逐周落地清单**，并且**严格贴合课程规范**（NeurIPS 模板、主文 4 页、1 页 proposal、8–10 分钟视频、硬性 DDL 等）。关键条款见课程 PDF：**NeurIPS 模板**与**4 页主文限制**、**需提交 Proposal/Report/Video**、**硬性截止时间 10/10 与 12/05**、**评分五维：Contribution/Clarity/Depth/Literature/Formatting（均值计分）**。  &#x20;

---

# 选题（定稿级别）

**题目**：**Bayes Routes：不确定性感知的稀疏 MoE 路由与稳健性评测**

**一句话贡献**：把 **MoE 路由器**从点估计变为**输入相关的后验 $q(\pi|x)$**（Concrete/Dirichlet 近似 + KL 先验 + 负载均衡），并给出**校准（ECE/NLL）+ OOD（AUROC/FPR95）+ 稳健性（CIFAR-C mCE）+ 计算开销**的统一小规模基准，展示在**几乎不掉主精度**前提下显著改善校准与 OOD。

**核心对标 Rubric**

* **Contribution**：通用的 *Bayesian 路由 for MoE* + 完整评测切面。
* **Clarity/Formatting**：NeurIPS 模板、4 页主文自洽，关键论断在主文内（附录可选）。
* **Depth/Literature**：系统消融（$\lambda,\tau,top\!-\!k$、推理策略、去 KL/去负载均衡）+ 近年 MoE/路由/校准文献对比。

---

# 10 周任务清单（覆盖 Proposal→Report→Video 全流程）

> **时间锚点**：**Proposal 截止 10/10 23:59**；**Report & Video 截止 12/05 23:59**（硬性）。

### W1（本周）：环境 & 数据 & 阅读卡（最小闭环）

* 复现环境：PyTorch + CIFAR-100/CIFAR-10、SVHN、CIFAR-C 数据管线；记录训练/评测脚本骨架。
* 跑通 **Dense ResNet-18** 小训练，打通 Acc/ECE/NLL/日志。
* 精读 3–4 篇核心（MoE 路由、Switch/Expert-Choice、Gumbel-Top-k、MC-Dropout），做 1 页术语表与差距定位。
* **产出**：`README` 初版 + 指标实现（ECE/NLL）+ 文献笔记（Related Work 雏形）。
* **Rubric 对齐**：Literature/Depth 开始积累。

### W2：标准稀疏 MoE 基线（top-k 路由 + 负载均衡）

* 在 ResNet 两个 MLP/FFN 位置植入 **MoE-MLP（4–8 专家, top-1/2）**；实现 **token dispatch/combine**、负载均衡项。
* 复现实验：CIFAR-100 主精度 + 负载直方图；建立**主表模板**（Dense vs MoE）。
* **产出**：基线表 0.1 版（Acc/NLL/ECE + FLOPs），可靠性图工具。
* **Rubric 对齐**：Formatting（图表/编号/脚注规范从现在就按 NeurIPS）。

### W3：**MC-Dropout 路由**（你给的建议里“先小步创新”）

* 在 **gating** 上加入 MC-Dropout：训练=常规；推理=多次前向，得到路由不确定性（熵/方差）。
* 评估 **校准提升**与**路由熵 vs 错误率**相关性；加入 **温度缩放**作对照。
* **产出**：对比曲线（ECE/NLL、可靠性图），学习率/温度/Dropout p 的小网格。
* **价值**：低改动→快速看到“不确定性信号”是否有效，为 W4–W6 的变分路由打底。

### W4：**Gumbel-Top-k** 训练路径 & 稳定性工程

* 实现 **Gumbel-Top-k/Concrete** 的可微近似（训练）+ 确定性 top-k（评估）；加入 **logit 范数 clip**、**梯度裁剪**。
* 完成 “**MoE baseline（softmax/noisy vs Gumbel-Top-k）**” 的等算力对比。
* **产出**：基线表 0.2 版 + 训练稳定性诊断（负载均衡热图、collapse 记录）。

### W5：**Bayesian 路由 v1（Concrete 后验 + KL 先验）**

* 让路由输出参数化 **$q(\pi|x)$**；目标：

  $$
  \mathcal L=\mathbb E_{\pi\sim q}[\text{CE}]+\lambda\,\mathrm{KL}(q\Vert p)+\beta\,\text{LoadBal}
  $$

  $p$ 取对称先验；温度从 0.7→0.5 线性降温；前 1–2 epoch 随机路由 warm-up。
* 跑通 **小网格**：$\lambda\in\{1e\!\!-\!\!4,1e\!\!-\!\!3,1e\!\!-\!\!2\}$，$\tau\in\{0.7,0.5,0.3\}$，top-k∈{1,2}。
* **产出**：主表 0.9 版（Dense / MoE / MC-Dropout-Gate / **Bayes-MoE**）。

### **W6：提交 Proposal（<= 1 页，NeurIPS 模板）& OOD/Corruption 首轮**

* 按课程要求写 **1 页 Proposal**（问题、假设、方法、数据与指标、风险），**10/10 23:59 前提交**。
* 完成 **CIFAR-C** 全套（mCE）与 **OOD（CIFAR-10↔SVHN）** AUROC/FPR95 评测；定位最佳 $\lambda,\tau,top-k$。
* **产出**：可靠性图、mCE 柱状、OOD ROC 曲线；**主结果图/表草稿**。
* **Rubric 对齐**：Proposal/Empirical 结构照课程建议组织。

### W7：**Compute-Adaptive（可选增强）** & 消融全开

* 用路由熵 **动态激活**：低熵→top-1，高熵→top-2；画 **精度/延迟** 前沿。
* 系统 **消融**：去 KL、去负载均衡、均值路由 vs 抽样路由、专家数 4 vs 8。
* **产出**：Ablation 表（影响量化），**Pareto 前沿图**；结论句式模板（“在等 FLOPs 下，ECE ↓X%”）。

### W8：写作冲刺（4 页主文必须自洽）

* 按课程 **Report 结构建议**起草：Abstract/Intro/Related/Method/Exp/Discussion/Conclusion。&#x20;
* 图表定稿 80%（方法图 1、主表 1、可靠性图 1、Pareto 或专家热图 1）。
* **附录**放实现细节/更多表格，但**主文必须完整**（评审不读附录是默认假设）。

### W9：复现实验包 & 排版/格式/引用

* 清理代码、固定随机种子、导出 `configs/*.yaml`、一键脚本；生成结果 CSV + 画图脚本。
* 严格检查 **NeurIPS 格式**、图表/编号/引用规范；按**五维评分**自检。
* **产出**：Report v1 完整 PDF + 可复现实验包（README 分步）。

### W10：最终结果 + 演示视频（8–10 分钟）

* 最后一轮对照（选 2–3 组关键超参重跑，锁定主表数值）；
* 录制 **8–10 分钟视频**（≤15 页 PDF 幻灯，图多字少，讲清问题/方法/结果/结论），按课程演示规范整理。&#x20;
* **12/05 23:59 前**同时提交 Report + Video + Slides。

---

## 交付物与打分映射（每周都在为“可评分证据”积累）

* **Proposal（1 页）**：现实且有雄心的目标、可验证假设、明确评测方案。
* **Report（主文 ≤4 页）**：主表 + 方法图 + 2 张关键图；所有主张有**引用/实验/或数学论证**支撑。&#x20;
* **Video（8–10 分钟）**：结构清晰、要点高亮、技术细节足量、图表专业。

---

## 风险与退路（Plan-B）

* **若 Bayes-MoE 收益不稳**：发布“**MoE vs 深度集成 vs MC-Dropout-Gate** 的等算力校准/OOD对比”作为研究结论，同样满足“**新洞见 + 完整评测**”的课程期待。
* **训练不稳/专家塌缩**：提高负载均衡权重、前 1–2 epoch 随机路由、logit clip 与梯度裁剪；先 top-1 再扩到 top-2。

---

如果你确认按这个 10 周表推进，我可以把 **W1–W2 的代码目录 & 类/函数签名清单**（`MoEBlock`、`BayesRouter(Concrete)`、`ece.py`、`ood_metrics.py`、`cifar_c_eval.py`、`train.yaml` 示例）直接给你，开箱即改。
