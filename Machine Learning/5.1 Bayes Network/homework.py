# %% [markdown]
# # 加载数据

# %%
import pandas as pd
pd.set_option('display.max_columns', None)
import sys, os
sys.path.insert(0, os.getcwd())

from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

# 加载数据
data = pd.read_csv('titanic.csv')

# 简单查看数据
data.info()
data.isnull().sum()
data['Fare'].describe()
data['Age'].describe()
data.mode().head(1)
data.head(5)

# %% [markdown]
# ## 简单的可视化进行分析

# %%
# 数据离散化
data["AgeGroup"] = pd.cut(
    data["Age"],
    bins=[0, 5, 14, 18, 30, 60, 100],
    labels=["Baby", "Child", "Teenager", "Adult", "OldAdult", "Old"],
)

data["FareGroup"] = pd.cut(
    data["Fare"],
    bins=[0, 15, 25, 50, 250, float("inf")],
    labels=["Low", "Middle", "High", "Expensive", "Luxury"],
)

survival_by_age_group = data.groupby('AgeGroup')['Survived'].mean()
survival_by_fare_group = data.groupby('FareGroup')['Survived'].mean()

print("Survival Rate by Age Group:\n", survival_by_age_group)
print("Survival Rate by Fare Group:\n", survival_by_fare_group)

# 创建一个包含2个子图的画布
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))


sns.barplot(x="AgeGroup", y="Survived", data=data, ax=axes[0], palette="coolwarm")
axes[0].set_title("Age Distribution")
axes[0].set_xlabel("Age Group")
axes[0].set_ylabel("Count")


sns.barplot(x="FareGroup", y="Survived", data=data, ax=axes[1], palette="coolwarm")

axes[1].set_title("Fare Distribution")
axes[1].set_xlabel("Fare Category")
axes[1].set_ylabel("Count")

plt.tight_layout()
plt.show()


# %% [markdown]
# # 特征处理
# ## 1、特征描述
# - PassengerId => 乘客ID
# 
# - Pclass => 客舱等级(1/2/3等舱位)
# 
# - Name => 乘客姓名
# 
# - Sex => 性别
# 
# - Age => 年龄 
# 
# - SibSp => 兄弟姐妹数/配偶数
# 
# - Parch => 父母数/子女数
# 
# - Ticket => 船票编号
# 
# - Fare => 船票价格 
# 
# - Cabin => 客舱号 
# 
# - Embarked => 登船港口 
# 
# ## 2、处理缺失值和离散值
# - Embarked 缺失值众数填充为S
# - 
# - Fare 缺失值中位数填充，划分为不同等级
# - 
# - Cabin 取第一个字母作为类别，缺失值用 'N' 填充
# - 
# - Age 划分为不同年龄段
# 

def preprocess_for_Bayesian(data_all):
    data = data_all.copy()
    # data["Embarked"].fillna(data["Embarked"].mode()[0], inplace=True)  # 用众数填充缺失值
    data["Fare"].fillna(data["Fare"].median(), inplace=True)  # 用中位数填充缺失值
    data.loc[data.Cabin.notnull(), "Cabin"] = 1
    data.loc[data.Cabin.isnull(), "Cabin"] = 0
    data.loc[data["Sex"] == "male", "Sex"] = 1
    data.loc[data["Sex"] == "female", "Sex"] = 0


    # 处理年龄特征
    data["Age"] = data["Age"].fillna(data["Age"].median())  # 填充缺失的年龄值
    data.loc[data["Age"] <= 18, "Age"] = 0  
    data.loc[(data["Age"] > 18) & (data["Age"] <= 35), "Age"] = 1  
    data.loc[data["Age"] > 35, "Age"] = 2  
    
    """
    Fare = data["Fare"].values
    Fare = Fare.reshape(-1, 1)
    km = KMeans(n_clusters=7)
    fare_fit = km.fit(Fare)
    
    print(fare_fit.labels_)
    # 可视化聚类
    plt.scatter(data["Fare"], data["Fare"], c=fare_fit.labels_, cmap="rainbow")
    plt.colorbar(label="Cluster Labels")
    plt.xlabel("Fare")
    plt.ylabel("Fare")
    plt.title("KMeans Clustering of Fare")
    plt.show()
    """
    
    # 处理票价特征
    data["Fare"] = data["Fare"].fillna(data["Fare"].median())
    data.loc[data["Fare"] <= 5, "Fare"] = 0
    data.loc[(data["Fare"] > 5) & (data["Fare"] <= 15), "Fare"] = 1
    data.loc[(data["Fare"] > 15) & (data["Fare"] <= 25), "Fare"] = 2
    data.loc[(data["Fare"] > 25) & (data["Fare"] <= 50), "Fare"] = 3
    data.loc[(data["Fare"] > 50) & (data["Fare"] <= 100), "Fare"] = 4
    data.loc[(data["Fare"] > 100) & (data["Fare"] <= 250), "Fare"] = 5
    data.loc[data["Fare"] > 250, "Fare"] = 6

    
    data["Cabin"] = data["Cabin"].astype(int)
    data["Fare"] = data["Fare"].astype(int)
    data["Age"] = data["Age"].astype(int)
    data["Sex"] = data["Sex"].astype(int)
    data["Pclass"] = data["Pclass"].astype(int)
    data["Survived"] = data["Survived"].astype(int)

    cols = ["Pclass", "Sex", "Age", "Fare", "Cabin", "Survived"]
    return data[cols]


preprocess_for_Bayesian(data)

# %%
from graphviz import Digraph

# 可视化贝叶斯网络模型
def showBN(model, save=False):
    """传入BayesianModel对象，调用graphviz绘制结构图，jupyter中可直接显示"""

    node_attr = dict(
        style="filled",
        shape="box",
        align="left",
        fontsize="12",
        ranksep="0.1",
        height="0.2",
        fillcolor="#B5C4DE", 
        color="#6B5B95",  
    )
    dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))
    seen = set()
    edges = model.edges()
    for a, b in edges:
        dot.edge(a, b,color='#A23B72')
    if save:
        dot.view(cleanup=True)
    return dot



# %% [markdown]
# # 设计贝叶斯网络(初步尝试)

# %%
from sklearn.model_selection import train_test_split
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator, BayesianEstimator
from pgmpy.inference import VariableElimination


# 划分数据集
all_data = preprocess_for_Bayesian(data)
X = all_data.iloc[:, :-1]
y = all_data.iloc[:, -1]
train, test= all_data.iloc[:int(len(all_data)*0.8),:], all_data.iloc[int(len(all_data)*0.8):,:]

# 设计贝叶斯网络（初步尝试）
model = BayesianNetwork()

edges = [
    ("Pclass", "Survived"),
    ("Sex", "Survived"),
    ("Age", "Survived"),
    ("Fare", "Survived"),
    ("Cabin", "Survived"),
]
model.add_edges_from(edges)


# 选取最佳模型结构
from pgmpy.estimators import HillClimbSearch
from pgmpy.estimators import BicScore

train = preprocess_for_Bayesian(data)
hc = HillClimbSearch(train)
best_model = hc.estimate(scoring_method=BicScore(train))
best_model = hc.estimate()




# %% [markdown]
# # 可视化贝叶斯网络

# %%
showBN(model) # 自己设计的
showBN(best_model) # 最佳模型结构
best_model.edges() # 最佳模型的边
best_model.nodes() # 最佳模型的节点

# %% [markdown]
# # 训练模型

# %%
# 贝叶斯估计

import json


best_1 = BayesianNetwork(best_model.edges())
best_1.fit(train, estimator=BayesianEstimator, prior_type="BDeu")

# 创建一个字典以保存所有 CPD
cpds = {}

# 打印所有变量的条件概率分布并将其保存为 JSON
for cpd in best_1.get_cpds():
    cpds[cpd.variable] = {
        "values": cpd.values.tolist(),  # 获取 CPD 的值
        "states": cpd.state_names,  # 获取 CPD 的状态名称
    }

# 将 CPD 字典保存为 JSON 文件
with open("cpd_data.json", "w") as json_file:
    json.dump(
        cpds,
        json_file,
        indent=4,
        default=lambda x: x.tolist() if hasattr(x, "tolist") else x,
    )

print("所有 CPD 已以 JSON 格式保存。")


# 预测
predict_data = test.drop("Survived", axis=1)
y_pred = best_1.predict(predict_data)

predicted_survived = y_pred["Survived"]
actual_survived = test["Survived"].reset_index(drop=True)
predicted_survived = predicted_survived.reset_index(drop=True)

# 计算预测准确率
accuracy = (predicted_survived == actual_survived).sum() / len(actual_survived)
print(f"预测准确率: {accuracy}")

# %%
# 最大似然估计
best_2 = BayesianNetwork(best_model.edges())
best_2.fit(train, estimator=MaximumLikelihoodEstimator)
predict_data = test.drop("Survived", axis=1)
y_pred = best_2.predict(predict_data)

predicted_survived = y_pred["Survived"]
actual_survived = test["Survived"].reset_index(drop=True)
predicted_survived = predicted_survived.reset_index(drop=True)

# 计算预测准确率
accuracy = (predicted_survived == actual_survived).sum() / len(actual_survived)
print(f"预测准确率: {accuracy}")

# %% [markdown]
# # MCMC 估计后验分布

# %%
from pgmpy.inference import VariableElimination
from pgmpy.models import BayesianModel
from pgmpy.sampling import BayesianModelSampling
from pgmpy.factors.discrete import TabularCPD

# 进行采样
sampling = BayesianModelSampling(best_1)
samples = sampling.forward_sample(size=1000)  # 采样1000个样本

# 分析后验分布
print(samples)


# %%
import matplotlib.pyplot as plt

cols = ["Pclass", "Sex", "Age", "Fare", "Cabin", "Survived"]

# 设置图形的行数和列数
n_cols = 3  # 每行显示 3 个图
n_rows = 2  # 每列显示 2 个图

# 创建子图
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 5))
axes = axes.flatten()  # 将多维数组扁平化以方便索引

# 绘制每个特征的后验分布
for idx, col in enumerate(cols):
    # 计算后验分布
    posterior_distribution = samples[col].value_counts(normalize=True)

    # 绘制条形图
    ax = axes[idx]
    posterior_distribution.plot(
        kind="bar",
        ax=ax,
        title=f"{col} Posterior Distribution",
        xlabel=col,
        ylabel="Probability",
        color="#008080",
        fontsize=15,
        rot=0
    )
    
    # 在每个条形上方显示数值
    for p in ax.patches:
        ax.text(p.get_x() + p.get_width() / 2., p.get_height(), 
                f'{p.get_height():.4f}', 
                ha='center', va='bottom', fontsize=12)
    # 增大坐标轴标签的字体大小
    ax.set_xlabel(ax.get_xlabel(), fontsize=15)
    ax.set_ylabel(ax.get_ylabel(), fontsize=15)
    ax.set_title(f"{col} Posterior Distribution", fontsize=15)  # 图标题的字体大小为 15
# 移除未使用的子图
for j in range(idx + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()  # 自动调整子图间距
plt.savefig("posterior_distribution.png", dpi=300)  # 保存图片
plt.show()


# %% [markdown]
# # 保存模型

# %%
best_1.save("best_model.bif")


