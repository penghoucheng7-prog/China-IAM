import pandas as pd
import numpy as np
import os, re, sys
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    r2_score, explained_variance_score,
    mean_squared_error, mean_absolute_error,
    median_absolute_error
)
import shap  # v0.41.0 及以上
import matplotlib.pyplot as plt

# ——— 全局设置 —————————————————————————————————————————————————————————
INPUT_PATH = r"C:\Users\phc\Desktop\中国模型比较\中国模型比较2\Middle_short_model_filtered_world_abbr_final.csv"
OUTPUT_DIR = r"C:\Users\phc\Desktop\中国模型比较\中国模型比较2\4_随机森林归因\二_归因处理\变量归因\results"
REGION_LIST = ["R10CHINA+", "CHN"]

def sanitize_tag(s: str) -> str:
    """将 Windows 路径不允许的字符替换为下划线。"""
    return re.sub(r'[<>:"/\\|?*]', '_', s)

def preprocess(target_variable: str):
    """将宽表展平成 Key-Year 样本，只剔除 Y 缺失，保留 X 中 NaN。"""
    df = pd.read_csv(INPUT_PATH, encoding="utf-8")
    df["Variable"] = df["Variable"].str.strip().str.lower()
    target = target_variable.strip().lower()

    # 年份列
    year_cols = [c for c in df.columns if c.isdigit() and 2010 <= int(c) <= 2050]
    if target not in df["Variable"].unique():
        raise ValueError(f"目标变量 `{target}` 不存在于 Variable 列")

    # 拆分 y 与 X
    y = df[df["Variable"] == target].copy()
    X = df[df["Variable"] != target].copy()

    # 构造 Key
    for col in ("Model", "Scenario", "Region"):
        if col not in df.columns:
            raise KeyError(f"缺少字段 `{col}`")
    for d in (y, X):
        d["Key"] = d["Model"] + "_" + d["Scenario"] + "_" + d["Region"]

    # 展开为长表
    y_long = y.melt(
        id_vars=["Key","Model","Scenario","Region"],
        value_vars=year_cols, var_name="Year", value_name="Y"
    )
    x_long = X.melt(
        id_vars=["Key","Model","Scenario","Region","Variable"],
        value_vars=year_cols, var_name="Year", value_name="X"
    )

    # pivot 特征
    x_pivot = x_long.pivot_table(
        index=["Key","Model","Scenario","Region","Year"],
        columns="Variable", values="X"
    ).reset_index()

    # 合并 + 仅剔除 Y 缺失
    merged = pd.merge(y_long, x_pivot,
                      on=["Key","Model","Scenario","Region","Year"], how="inner")
    merged = merged.dropna(subset=["Y"])

    feat_cols = [c for c in merged.columns
                 if c not in ["Key","Model","Scenario","Region","Year","Y"]]
    X_clean = merged[feat_cols]
    y_clean = merged["Y"].values
    return X_clean, y_clean, merged

def train_full_model(X, y):
    """训练全样本随机森林模型，限制深度与并行数防止内存爆炸 :contentReference[oaicite:5]{index=5}。"""
    model = RandomForestRegressor(
        n_estimators=80, max_depth=10,
        max_features="sqrt", n_jobs=1,  # 限制并行线程
        random_state=42, warm_start=True
    )
    model.fit(X, y)
    return model

def finetune_by_region(model, merged, X, y):
    """对指定区域子样本微调。"""
    mask = merged["Region"].isin(REGION_LIST)
    X_sub, y_sub = X.loc[mask.values], y[mask.values]
    print(f">>> 地区子样本数量：{len(X_sub)}")
    model.set_params(n_estimators=model.n_estimators + 40)
    model.fit(X_sub, y_sub)
    return model, X_sub, y_sub

def evaluate_and_save(model, X_sub, y_sub, tag, target):
    """评估并保存指标、特征重要性、SHAP 图表。"""
    Xtr, Xte, ytr, yte = train_test_split(
        X_sub, y_sub, test_size=0.3, random_state=42
    )
    ytr_pred, yte_pred = model.predict(Xtr), model.predict(Xte)
    metrics = {
        "R2_in": r2_score(ytr, ytr_pred),
        "R2_out": r2_score(yte, yte_pred),
        "EVS_out": explained_variance_score(yte, yte_pred),
        "MSE_out": mean_squared_error(yte, yte_pred),
        "MAE_out": mean_absolute_error(yte, yte_pred),
        "MedAE_out": median_absolute_error(yte, yte_pred)
    }
    safe = sanitize_tag(tag)
    out_dir = os.path.join(OUTPUT_DIR, safe)
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(
        os.path.join(out_dir, f"{safe}_模型指标.csv"),
        index=False, encoding="utf-8-sig"
    )

    # 特征相对重要性（前10）
    rel_imp = model.feature_importances_ / model.feature_importances_.sum() * 100
    imp_df = pd.DataFrame({
        "变量": X_sub.columns,
        "相对重要性(%)": np.round(rel_imp,4)
    }).nlargest(10, "相对重要性(%)")
    imp_df.to_csv(
        os.path.join(out_dir, f"{safe}_变量相对重要性.csv"),
        index=False, encoding="utf-8-sig"
    )

    # SHAP 解释器：使用自动模式或提供背景数据防止无背景下崩溃 :contentReference[oaicite:6]{index=6}
    background = shap.kmeans(X_sub, n_clusters=50)  # 聚类选择代表性背景
    explainer = shap.TreeExplainer(model, data=background, feature_perturbation="interventional")
    sample = X_sub.iloc[np.random.choice(X_sub.shape[0], size=min(200, X_sub.shape[0]), replace=False)]
    shap_vals = explainer.shap_values(sample)  # 不再使用 deprecated 参数

    # 绘制带均值黑线的 SHAP Summary
    shap.summary_plot(shap_vals, sample, show=False)
    fig = plt.gcf(); ax = fig.axes[0]
    ax.set_xlabel(""); ax.set_ylabel("")
    cbar = fig.axes[-1]
    cbar.set_ylabel(""); cbar.set_yticklabels([]); cbar.set_xticklabels([])

    mean_shap = np.abs(shap_vals).mean(axis=0)
    ordered = np.argsort(mean_shap)[::-1]
    for i, idx in enumerate(ordered):
        mv = mean_shap[idx]
        ax.hlines(y=i, xmin=mv-0.01, xmax=mv+0.01,
                  color="black", linewidth=12, zorder=20)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{safe}_SHAP.png"), dpi=300)
    plt.close()

    shap_df = pd.DataFrame({
        "变量": sample.columns,
        "SHAP均值": np.round(mean_shap,6)
    }).sort_values("SHAP均值", ascending=False)
    shap_df.to_csv(
        os.path.join(out_dir, f"{safe}_mean_SHAP.csv"),
        index=False, encoding="utf-8-sig"
    )

    print(f"[✔] `{tag}` 结果已保存至：{out_dir}")

def main(target_variable: str):
    """主流程：预处理 → 全样本训练 → 地区微调 → 保存结果。"""
    print(f"→ 分析目标：{target_variable}")
    X, y, merged = preprocess(target_variable)
    print(f"→ 样本量 (Key-Year)：{len(X)}")

    full_model = train_full_model(X, y)
    print("→ 全样本训练完成")

    tuned_model, X_sub, y_sub = finetune_by_region(full_model, merged, X, y)
    print("→ 地区微调完成")

    tag = f"{target_variable}"
    evaluate_and_save(tuned_model, X_sub, y_sub, tag, target_variable)

if __name__ == "__main__":
    if len(sys.argv)==2:
        main(sys.argv[1])
    else:
        print("用法: python script.py <目标变量>") 