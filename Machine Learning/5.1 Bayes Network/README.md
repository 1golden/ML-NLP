

# 贝叶斯推理在复杂系统建模中的应用实验报告

本目录包含与实验相关的代码、数据、结果以及模型文件，详细描述了基于贝叶斯网络的推理与分析过程，以及结果的可视化。

---

## 文件说明

### 数据文件
- **`titanic.csv`**
  - 内容：实验数据集，基于泰坦尼克号乘客信息，包括乘客的生存情况、社会阶层（Pclass）、性别（Sex）等特征。
  - 用途：用于构建贝叶斯网络和后续的推理分析。

---

### 模型文件
- **`best_model.bif`**
  - 内容：保存了实验中训练得到的贝叶斯网络模型，格式为 BIF（Bayesian Interchange Format）。
  - 用途：模型文件可用于加载和推断贝叶斯网络结构及条件概率分布。

---

### JSON 数据
- **`cpd_data.json`**
  - 内容：保存了贝叶斯网络中各节点的条件概率分布（CPD），以 JSON 格式存储。
  - 用途：记录了每个节点的概率分布，用于进一步的分析和可视化。

---

### 可视化输出
- **`Digraph.gv.pdf`**
  - 内容：生成的贝叶斯网络拓扑结构图，以 PDF 格式存储。
  - 用途：展示各特征之间的因果关系及网络结构。
  
- **`MCMC_posterior_distribution.png`**
  - 内容：基于 MCMC 方法生成的后验分布图。
  - 用途：展示后验分布的采样结果，分析系统中参数的不确定性。

- **`Survival_Probability_Pclass_Sex.png`**
  - 内容：展示特征 `Pclass`（乘客社会阶层）和 `Sex`（性别）对生存概率的影响。
  - 用途：用于探讨不同特征对乘客生存情况的作用。

---

### 代码文件
- **`homework.ipynb`**
  - 内容：实验的完整代码，基于 Jupyter Notebook 编写。
  - 用途：实现贝叶斯网络建模、MCMC 推断以及结果的可视化等全过程。

---

## 主要实验内容
1. **贝叶斯网络建模**：
   - 通过泰坦尼克号数据集，构建以 `Survived`（生存）为目标变量的贝叶斯网络。
   - 结合先验知识设置网络拓扑结构，并使用贝叶斯估计器（Bayesian Estimator）计算条件概率分布。

2. **后验分布推断**：
   - 使用 MCMC 方法采样目标变量的后验分布。
   - 分析关键特征（如 `Pclass`、`Sex`）对 `Survived` 的影响。

3. **结果可视化**：
   - 网络拓扑图（`Digraph.gv.pdf`）展示特征间的因果关系。
   - 条件概率分布图（`Survival_Probability_Pclass_Sex.png`）量化特征对生存概率的影响。
   - MCMC 后验分布图（`MCMC_posterior_distribution.png`）分析参数的不确定性。

---

## 使用说明
1. **加载模型**：
   - 使用 `best_model.bif` 重新加载贝叶斯网络。
2. **运行代码**：
   - 打开 `homework.ipynb`，按照单元格顺序运行以复现实验结果。
3. **可视化分析**：
   - 查看 `Survival_Probability_Pclass_Sex.png` 和 `MCMC_posterior_distribution.png` 以获取实验结果的可视化。

---

## 实验依赖
- **Python 版本**：3.8+
- **主要库**：
  - `pgmpy`
  - `pandas`
  - `matplotlib`
  - `seaborn`

---

如果有进一步问题，请联系实验维护者或参考 `homework.ipynb` 中的详细注释。

---