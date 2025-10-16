import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold, cross_val_predict, GridSearchCV, train_test_split, learning_curve
from sklearn.inspection import permutation_importance
import seaborn as sns
import matplotlib.pyplot as plt
import shap
import warnings
warnings.filterwarnings("ignore")

# --------------------------------
# 数据加载
# --------------------------------
file_path = r"F:\RRFF611.xlsx"
df_raw = pd.read_excel(file_path)
df_raw = df_raw.rename(columns={
    '电压': 'Voltage',
    '360捕获效率': '360捕获效率',
    '390捕获效率': '390捕获效率',
    '420捕获效率': '420捕获效率',
    '360富集率': '360富集率',
    '390富集率': '390富集率',
    '420富集率': '420富集率'
})
st.write("✅ Data loaded successfully:", df_raw.head())

records = []
for _, row in df_raw.iterrows():
    for flow in [360, 390, 420]:
        capture = row[f'{flow}捕获效率']
        enrich = row[f'{flow}富集率']
        records.append([row['Voltage'], flow, enrich, capture])

df = pd.DataFrame(records, columns=['Voltage', 'Flow Rate', 'Enrichment Ratio', 'Capture Efficiency'])
df['Diameter Threshold'] = np.linspace(15.6, 16.6, len(df))

X = df[['Voltage', 'Flow Rate', 'Diameter Threshold']]
y_enrich = df['Enrichment Ratio']
y_capture = df['Capture Efficiency']

# --------------------------------
# 模型构建函数
# --------------------------------
def build_rf():
    return RandomForestRegressor(
        n_estimators=200,        # 减少树的数量
        max_features='sqrt',
        max_depth=8,             # 限制深度
        min_samples_split=5,     # 增加分裂所需最小样本数
        min_samples_leaf=3,      # 关键：增加叶节点最小样本数
        bootstrap=True,
        random_state=42
    )

model_enrich = build_rf()
model_capture = build_rf()

cv = KFold(n_splits=5, shuffle=True, random_state=42)
y_enrich_pred = cross_val_predict(model_enrich, X, y_enrich, cv=cv)
y_capture_pred = cross_val_predict(model_capture, X, y_capture, cv=cv)
model_enrich.fit(X, y_enrich)
model_capture.fit(X, y_capture)

# --------------------------------
# 性能指标
# --------------------------------
def metrics(y_true, y_pred):
    return r2_score(y_true, y_pred), np.sqrt(mean_squared_error(y_true, y_pred)), mean_absolute_error(y_true, y_pred)


enrich_r2_train = r2_score(y_enrich, model_enrich.predict(X))
capture_r2_train = r2_score(y_capture, model_capture.predict(X))
enrich_r2_cv, _, _ = metrics(y_enrich, y_enrich_pred)
capture_r2_cv, _, _ = metrics(y_capture, y_capture_pred)

st.subheader("Model Performance Comparison")
st.markdown(f"**Enrichment Ratio:** 训练 R² = {enrich_r2_train:.3f} | 交叉验证 R² = {enrich_r2_cv:.3f}")
st.markdown(f"**Capture Efficiency:** 训练 R² = {capture_r2_train:.3f} | 交叉验证 R² = {capture_r2_cv:.3f}")

# =====================================================
# 🧩 1️⃣ 嵌套交叉验证
# =====================================================
def nested_cv_evaluation(X, y):
    outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
    inner_cv = KFold(n_splits=3, shuffle=True, random_state=42)
    param_grid = {'max_depth': [5, 10, 15, None], 'n_estimators': [300, 500, 800]}
    outer_scores = []

    for train_idx, test_idx in outer_cv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        grid = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=inner_cv, scoring='r2', n_jobs=-1)
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_

        outer_r2 = r2_score(y_test, best_model.predict(X_test))
        outer_scores.append(outer_r2)

    return np.mean(outer_scores), np.std(outer_scores)

enrich_nested_mean, enrich_nested_std = nested_cv_evaluation(X, y_enrich)
capture_nested_mean, capture_nested_std = nested_cv_evaluation(X, y_capture)
st.markdown(f"**嵌套交叉验证 R²（Enrichment）:** {enrich_nested_mean:.3f} ± {enrich_nested_std:.3f}")
st.markdown(f"**嵌套交叉验证 R²（Capture）:** {capture_nested_mean:.3f} ± {capture_nested_std:.3f}")

# =====================================================
# 📈 2️⃣ 学习曲线
# =====================================================
def plot_learning_curve(model, X, y, title):
    train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=10, scoring='r2', n_jobs=-1)
    train_mean, test_mean = np.mean(train_scores, axis=1), np.mean(test_scores, axis=1)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(train_sizes, train_mean, 'o-', label='Training R²')
    ax.plot(train_sizes, test_mean, 'o-', label='Validation R²')
    ax.set_title(title)
    ax.set_xlabel("Training Samples")
    ax.set_ylabel("R² Score")
    ax.legend()
    st.pyplot(fig)

plot_learning_curve(model_enrich, X, y_enrich, "Learning Curve: Enrichment")
plot_learning_curve(model_capture, X, y_capture, "Learning Curve: Capture")

# =====================================================
# 🔍 3️⃣ Permutation & SHAP 重要性
# =====================================================
def plot_permutation_importance(model, X, y, title):
    result = permutation_importance(model, X, y, n_repeats=10, random_state=42)
    imp_df = pd.DataFrame({'Feature': X.columns, 'Importance': result.importances_mean})
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.barplot(x='Importance', y='Feature', data=imp_df, ax=ax)
    ax.set_title(title)
    st.pyplot(fig)

st.subheader("Permutation Importance")
plot_permutation_importance(model_enrich, X, y_enrich, "Permutation Importance: Enrichment")
plot_permutation_importance(model_capture, X, y_capture, "Permutation Importance: Capture")

# ---- SHAP ----
st.subheader("SHAP Feature Contributions")
explainer = shap.TreeExplainer(model_enrich)
shap_values = explainer.shap_values(X)
fig, ax = plt.subplots(figsize=(6, 5))
shap.summary_plot(shap_values, X, plot_type="bar", show=False)
st.pyplot(fig)

# =====================================================
# 🚀 4️⃣ Out-of-Distribution Test（未见组合）
# =====================================================
def ood_test(model, df, unseen_voltage=480, unseen_flow=450):
    test_X = pd.DataFrame({
        'Voltage': [unseen_voltage] * 5,
        'Flow Rate': np.linspace(350, 450, 5),
        'Diameter Threshold': np.linspace(15.5, 16.5, 5)
    })
    preds = model.predict(test_X)
    test_X['Predicted'] = preds
    return test_X

st.subheader("Out-of-Distribution Test (Unseen Parameters)")
ood_enrich = ood_test(model_enrich, df)
ood_capture = ood_test(model_capture, df)
st.write("**Enrichment Predictions:**", ood_enrich)
st.write("**Capture Predictions:**", ood_capture)
