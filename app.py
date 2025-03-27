# -*- coding: utf-8 -*-
import streamlit as st
import joblib
import numpy as np
import shap
import matplotlib.pyplot as plt

# 设置matplotlib支持中文和负号
plt.rcParams['font.sans-serif'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

# 加载模型
pg_model_path = "pg.pkl"
pn_model_path = "pn.pkl"
pg_model = joblib.load(pg_model_path)
pn_model = joblib.load(pn_model_path)

# 特征定义
pg_features = ['LVEF', 'BNP', 'CK', 'CRP', 'Cre', 'HbAlc', 'LD', 'TG', 'LVIDd',
               'LVIDs', 'E_', 'etoa', 'dct', 'heartrate', 'printerval', 'rv5',
               'rv6', 'rv5sv1', 'rv6sv1']

pn_features = ['LVEF', 'ALP', 'ALT', 'BUN', 'CK', 'GLU', 'Hb', 'Hct', 'MCH',
               'MCV', 'PLT', 'TBIL', 'UA', 'LVIDd', 'LVIDs', 'etoa', 'dct',
               'age', 'bmi', 'heartrate', 'printerval', 'pwaveaxis',
               'qrsaxis', 'twaveaxis', 'rv5', 'rv6', 'rv6sv1']

# 设置页面配置
st.set_page_config(layout="wide", page_title="PCI术后LVEF恢复预测", page_icon="❤️")
st.title("❤️ PCI术后LVEF恢复预测 (Preserved EF)")

# 动态生成输入项
st.sidebar.header("特征输入区域")
st.sidebar.write("请输入特征值：")

# 创建输入字典
input_features = {}

# 添加PG模型特征输入
st.sidebar.subheader("Good Recovery 特征")
for feature in pg_features:
    input_features[feature] = st.sidebar.number_input(
        label=f"{feature}",
        value=0.0,
        format="%.2f"
    )

# 添加PN模型特征输入
st.sidebar.subheader("Normal Recovery 特征")
for feature in set(pn_features) - set(pg_features):
    input_features[feature] = st.sidebar.number_input(
        label=f"{feature}",
        value=0.0,
        format="%.2f"
    )

# 预测与概率计算
if st.button("预测恢复状态"):
    # 准备特征向量
    pg_feature_vector = np.array([input_features[f] for f in pg_features]).reshape(1, -1)
    pn_feature_vector = np.array([input_features[f] for f in pn_features]).reshape(1, -1)

    # 获取每个模型的预测概率
    pg_prob = pg_model.predict_proba(pg_feature_vector)[0][1]  # 假设第二类是positive
    pn_prob = pn_model.predict_proba(pn_feature_vector)[0][1]

    # 计算非恢复概率
    non_recovery_prob = 1 - (pg_prob + pn_prob)

    # 展示预测结果
    st.write("### 预测结果")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Good Recovery", f"{pg_prob:.2%}")
    with col2:
        st.metric("Normal Recovery", f"{pn_prob:.2%}")
    with col3:
        st.metric("Non Recovery", f"{non_recovery_prob:.2%}")

    # SHAP解释 (可选)
    st.write("### 特征重要性解释")

    # 对PG模型的SHAP解释
    pg_explainer = shap.TreeExplainer(pg_model)
    pg_shap_values = pg_explainer.shap_values(pg_feature_vector)

    # 绘制PG模型的SHAP力图
    plt.figure(figsize=(10, 5))
    shap.force_plot(
        pg_explainer.expected_value[1],
        pg_shap_values[1][0],
        pg_feature_vector[0],
        matplotlib=True,
        show=False
    )
    plt.title("Good Recovery 特征重要性")
    st.pyplot(plt)

    # 对PN模型的SHAP解释
    pn_explainer = shap.TreeExplainer(pn_model)
    pn_shap_values = pn_explainer.shap_values(pn_feature_vector)

    plt.figure(figsize=(10, 5))
    shap.force_plot(
        pn_explainer.expected_value[1],
        pn_shap_values[1][0],
        pn_feature_vector[0],
        matplotlib=True,
        show=False
    )
    plt.title("Normal Recovery 特征重要性")
    st.pyplot(plt)

# # 添加使用说明
# st.sidebar.write("""
# ### 使用说明
# 1. 在左侧输入所有特征值
# 2. 点击"预测恢复状态"
# 3. 查看三种恢复状态的概率
# 4. 查看特征重要性解释
# """)