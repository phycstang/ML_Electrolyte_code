"""
本脚本在原有管线基础上，新增：
1) 在预处理后导出一份整体“已归一化”的特征副本 processed_data_scaled.csv （# NEW）
2) 保存缩放器统计量 scaler_stats.npz 方便复现（# NEW）

训练阶段仍通过 Pipeline 中的 StandardScaler 严格在 CV 内拟合，避免数据泄漏（符合 sklearn 推荐实践）。
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    ExtraTreesClassifier, AdaBoostClassifier,
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.datasets import make_classification

try:
    from lightgbm import LGBMClassifier
    _lightgbm_available = True
except ImportError:
    LGBMClassifier = None  # type: ignore
    _lightgbm_available = False

try:
    from catboost import CatBoostClassifier
    _catboost_available = True
except ImportError:
    CatBoostClassifier = None  # type: ignore
    _catboost_available = False

from xgboost import XGBClassifier

warnings.filterwarnings('ignore')


def load_or_generate_dataset(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        df = pd.read_csv(path)
        print(f"Loaded dataset from {path} with shape {df.shape}.")
    else:
        print(f"Dataset file {path} not found; generating a synthetic dataset for demonstration.")
        X_syn, y_syn = make_classification(
            n_samples=500, n_features=50, n_informative=30, n_redundant=10, n_classes=3, random_state=42
        )
        columns = [f"feature_{i}" for i in range(X_syn.shape[1])]
        df = pd.DataFrame(X_syn, columns=columns)
        df['id_dim'] = 0
        df['id_connect'] = 0
        df['id_cif'] = 0
        df['id_cif_path'] = 0
        df['id_score'] = y_syn
        print(f"Generated synthetic dataset with shape {df.shape}.")
    return df


def preprocess_data(df: pd.DataFrame):
    value_counts = df['id_score'].value_counts()
    rare_classes = value_counts[value_counts < 3].index
    if len(rare_classes) > 0:
        df = df[~df['id_score'].isin(rare_classes)]
        print(f"Removed {len(rare_classes)} rare class(es).")

    le = LabelEncoder()
    y_encoded = le.fit_transform(df['id_score'])

    drop_cols = [c for c in ['id_score','id_dim','id_connect','id_cif','id_cif_path'] if c in df.columns]
    X = df.drop(columns=drop_cols)

    # 仅保留数值列，防止意外的非数值列（# NEW：稳妥起见）
    X = X.select_dtypes(include=[np.number])

    const_cols = [c for c in X.columns if X[c].nunique() == 1]
    if const_cols:
        X = X.drop(columns=const_cols)
        print(f"Dropped {len(const_cols)} constant column(s).")

    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    threshold = 0.95
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    if to_drop:
        X = X.drop(columns=to_drop)
        print(f"Dropped {len(to_drop)} highly correlated column(s) with threshold {threshold}.")

    return X, pd.Series(y_encoded, index=X.index), le


def visualize_correlation_distribution(upper: pd.DataFrame, threshold: float) -> None:
    corr_values = upper.abs().stack().values
    plt.figure(figsize=(8, 5))
    plt.hist(corr_values, bins=50)
    plt.axvline(x=threshold, linestyle='--', label=f'Threshold {threshold}')
    plt.title('Distribution of Absolute Correlation Values')
    plt.xlabel('Absolute Correlation')
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()
    plt.savefig('correlation_distribution.png', dpi=300)
    plt.close()


def build_models() -> dict:
    models: dict[str, dict] = {
        'LogisticRegression': {
            'model': LogisticRegression(max_iter=300),
            'params': {'model__C': [0.1, 1.0]},
        },
        'RandomForest': {
            'model': RandomForestClassifier(random_state=42),
            'params': {'model__n_estimators': [100, 200]},
        },
        'GradientBoosting': {
            'model': GradientBoostingClassifier(random_state=42),
            'params': {'model__learning_rate': [0.05, 0.1]},
        },
        'SVM': {
            'model': SVC(probability=False),
            'params': {'model__C': [0.1, 1.0]},
        },
        'KNN': {
            'model': KNeighborsClassifier(),
            'params': {'model__n_neighbors': [3, 5]},
        },
        'ExtraTrees': {
            'model': ExtraTreesClassifier(random_state=42),
            'params': {'model__n_estimators': [100, 150]},
        },
        'AdaBoost': {
            'model': AdaBoostClassifier(random_state=42),
            'params': {'model__n_estimators': [50, 100]},
        },
        'GaussianNB': {
            'model': GaussianNB(),
            'params': {},
        },
        'MLP': {
            'model': MLPClassifier(max_iter=300, random_state=42),
            'params': {'model__hidden_layer_sizes': [(50,), (100,)]},
        },
        'XGBoost': {
            'model': XGBClassifier(
                objective='multi:softprob', eval_metric='mlogloss',
                use_label_encoder=False, random_state=42
            ),
            'params': {'model__n_estimators': [100]},
        },
        'DecisionTree': {
            'model': DecisionTreeClassifier(random_state=42),
            'params': {'model__max_depth': [None, 10]},
        },
        'LinearDiscriminantAnalysis': {
            'model': LinearDiscriminantAnalysis(),
            'params': {},
        },
        'QuadraticDiscriminantAnalysis': {
            'model': QuadraticDiscriminantAnalysis(),
            'params': {},
        },
    }
    if _lightgbm_available and LGBMClassifier is not None:
        models['LightGBM'] = {
            'model': LGBMClassifier(random_state=42, verbose=-1),
            'params': {'model__n_estimators': [100]},
        }
    if _catboost_available and CatBoostClassifier is not None:
        models['CatBoost'] = {
            'model': CatBoostClassifier(random_state=42, verbose=0, loss_function='MultiClass'),
            'params': {'model__iterations': [100]},
        }
    return models


def plot_param_search_results(cv_results: dict, model_name: str) -> None:
    results_df = pd.DataFrame(cv_results)
    if 'mean_test_score' not in results_df:
        return
    mean_scores = results_df['mean_test_score']
    param_cols = [col for col in results_df.columns if col.startswith('param_')]
    for param_col in param_cols:
        param_name = param_col.replace('param_model__', '').replace('param_', '')
        param_values = results_df[param_col].unique()
        x_vals, y_vals = [], []
        for val in param_values:
            mask = results_df[param_col] == val
            mean_score = mean_scores[mask].mean()
            x_vals.append(str(val))
            y_vals.append(mean_score)
        if len(x_vals) < 2:
            continue
        plt.figure(figsize=(7, 5))
        plt.plot(x_vals, y_vals, marker='o')
        plt.title(f'Parameter Tuning for {model_name}: {param_name}')
        plt.xlabel(param_name)
        plt.ylabel('Mean CV Accuracy')
        plt.tight_layout()
        fname = f'{model_name}_{param_name}_tuning.png'
        plt.savefig(fname, dpi=300)
        plt.close()


def main() -> None:
    path = 'extra_features_f.csv'
    df = load_or_generate_dataset(path)

    # 预处理
    X, y_encoded, label_encoder = preprocess_data(df)

    # ---- 保存未缩放的处理后数据（保持原逻辑） ----
    processed_data = pd.concat(
        [X.reset_index(drop=True), df.loc[X.index, 'id_score'].reset_index(drop=True)], axis=1
    )
    processed_data.to_csv('processed_data.csv', index=False)
    print(f"Processed data saved to 'processed_data.csv' with shape {processed_data.shape}.")

    # ---- 导出“整体已归一化”的副本（# NEW）----
    scaler_export = StandardScaler()                  # Z-score 标准化
    X_scaled_export = scaler_export.fit_transform(X)  # 注意：仅用于导出副本，不参与训练（训练仍在 Pipeline 中缩放）
    processed_scaled = pd.concat(
        [pd.DataFrame(X_scaled_export, columns=X.columns, index=X.index).reset_index(drop=True),
         df.loc[X.index, 'id_score'].reset_index(drop=True)], axis=1
    )
    processed_scaled.to_csv('processed_data_scaled.csv', index=False)
    # 保存统计量，便于外部复现： z = (x - mean) / scale
    np.savez('scaler_stats.npz', mean=scaler_export.mean_, scale=scaler_export.scale_)
    print("Scaled copy saved to 'processed_data_scaled.csv' and scaler stats saved to 'scaler_stats.npz'.")

    # 相关性分布（Pearson 对缩放不敏感，因此继续用未缩放 X 计算）
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    visualize_correlation_distribution(upper, threshold=0.95)
    print("Correlation distribution plot saved to 'correlation_distribution.png'.")

    # 划分数据
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # 模型与参数
    models = build_models()
    cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
    results: list[tuple[str, dict, float, float]] = []

    # 逐模型训练
    for name, mp in models.items():
        print(f"Training {name}...")
        pipe = Pipeline([
            ('scaler', StandardScaler()),  # 训练中在 Pipeline 内做标准化，避免泄漏（# CHANGED：强调）
            ('model', mp['model'])
        ])
        if mp['params']:
            grid = GridSearchCV(
                pipe, mp['params'], scoring='accuracy', cv=cv,
                n_jobs=1, error_score='raise'
            )
            grid.fit(X_train, y_train)
            best_model = grid.best_estimator_
            best_params = grid.best_params_
            print(f"Best params for {name}: {best_params}")
            try:
                plot_param_search_results(grid.cv_results_, name)
            except Exception as e:
                print(f"Could not plot tuning curves for {name}: {e}")
        else:
            pipe.fit(X_train, y_train)
            best_model = pipe
            best_params = {}

        y_pred = best_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        results.append((name, best_params, acc, f1))
        print(f"{name}: Accuracy={acc:.4f}, F1_macro={f1:.4f}")

    res_df = pd.DataFrame(results, columns=['Model', 'Best_Params', 'Test_Accuracy', 'Test_F1_macro'])
    res_df.sort_values('Test_F1_macro', ascending=False, inplace=True)
    res_df.to_csv('model_results.csv', index=False)
    print("Model results saved to 'model_results.csv'.")

    # 结果可视化
    plt.figure(figsize=(12, 6))
    plt.bar(res_df['Model'], res_df['Test_Accuracy'])
    plt.ylim(0, 1)
    plt.xticks(rotation=45, ha='right')
    plt.title('Model Accuracy')
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.tight_layout()
    plt.savefig('extended_model_accuracy.png', dpi=300)
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.bar(res_df['Model'], res_df['Test_F1_macro'])
    plt.ylim(0, 1)
    plt.xticks(rotation=45, ha='right')
    plt.title('Model F1 Macro')
    plt.xlabel('Model')
    plt.ylabel('F1 Macro')
    plt.tight_layout()
    plt.savefig('extended_model_f1.png', dpi=300)
    plt.close()
    print("Model accuracy and F1 bar charts saved.")

    # 特征重要性（树模型），保持训练集标准化的一致性
    top_models = res_df.head(2)['Model'].tolist()
    print(f"Top two models by F1_macro: {top_models}")

    for model_name in res_df['Model']:
        if model_name not in ['RandomForest','ExtraTrees','DecisionTree','XGBoost','LightGBM','CatBoost']:
            continue

        if model_name == 'RandomForest':
            clf = RandomForestClassifier(random_state=42)
        elif model_name == 'ExtraTrees':
            clf = ExtraTreesClassifier(random_state=42)
        elif model_name == 'DecisionTree':
            clf = DecisionTreeClassifier(random_state=42)
        elif model_name == 'XGBoost':
            clf = XGBClassifier(
                objective='multi:softprob', eval_metric='mlogloss',
                use_label_encoder=False, random_state=42
            )
        elif model_name == 'LightGBM' and _lightgbm_available:
            clf = LGBMClassifier(random_state=42, verbose=-1)
        elif model_name == 'CatBoost' and _catboost_available:
            clf = CatBoostClassifier(random_state=42, verbose=0, loss_function='MultiClass')
        else:
            continue

        best_params = res_df[res_df['Model']==model_name].iloc[0]['Best_Params']
        # 去掉 'model__' 前缀
        param_clean = {}
        for k, v in best_params.items():
            if '__' in k:
                param_clean[k.split('__')[1]] = v

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)  # 与训练阶段一致（# CHANGED：显式说明）
        try:
            clf.set_params(**param_clean)
        except Exception:
            pass
        try:
            clf.fit(X_train_scaled, y_train)
        except Exception as e:
            print(f"Could not fit {model_name} for feature importance: {e}")
            continue

        if hasattr(clf, 'feature_importances_'):
            importances = clf.feature_importances_
            top_n = 15
            indices = np.argsort(importances)[::-1][:min(len(importances), top_n)]
            features = X.columns
            top_features = [features[i] for i in indices]
            top_importances = importances[indices]
            plt.figure(figsize=(10, 6))
            plt.barh(range(len(top_features)), top_importances)
            plt.yticks(range(len(top_features)), top_features)
            plt.gca().invert_yaxis()
            plt.title(f'Top {len(top_features)} Feature Importances ({model_name})')
            plt.xlabel('Importance')
            plt.ylabel('Feature')
            plt.tight_layout()
            fname = f'{model_name}_feature_importance.png'
            plt.savefig(fname, dpi=300)
            plt.close()
            print(f"Saved feature importance plot for {model_name} to '{fname}'.")

    # PCA（全量 X 做标准化）
    scaler_all = StandardScaler()
    X_scaled_all = scaler_all.fit_transform(X)
    pca = PCA(n_components=2, random_state=42)
    components = pca.fit_transform(X_scaled_all)
    plt.figure(figsize=(10, 7))
    unique_labels = np.unique(y_encoded)
    for lbl in unique_labels:
        mask = y_encoded == lbl
        label_name = label_encoder.inverse_transform([lbl])[0]
        plt.scatter(components[mask, 0], components[mask, 1], label=str(label_name), s=40)
    plt.title('PCA Scatter Plot (2 components)')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(title='id_score')
    plt.tight_layout()
    plt.savefig('pca_scatter.png', dpi=300)
    plt.close()
    print("PCA scatter plot saved.")

    # 最佳模型混淆矩阵
    best_model_name = res_df.iloc[0]['Model']
    best_params_dict = res_df.iloc[0]['Best_Params']
    best_mp = build_models()[best_model_name]  # 重新实例化以确保干净
    best_pipe = Pipeline([('scaler', StandardScaler()), ('model', best_mp['model'])])
    if best_params_dict:
        best_pipe.set_params(**best_params_dict)
    best_pipe.fit(X_train, y_train)
    best_pred = best_pipe.predict(X_test)

    true_labels = label_encoder.inverse_transform(y_test)
    pred_labels = label_encoder.inverse_transform(best_pred)
    unique_sorted = sorted(np.unique(true_labels))
    cm = confusion_matrix(true_labels, pred_labels, labels=unique_sorted)

    plt.figure(figsize=(10, 8))
    plt.imshow(cm)
    plt.title(f'Confusion Matrix - Best Model ({best_model_name})')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(ticks=range(len(unique_sorted)), labels=unique_sorted, rotation=45, ha='right')
    plt.yticks(ticks=range(len(unique_sorted)), labels=unique_sorted)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha='center', va='center')
    plt.tight_layout()
    plt.savefig('extended_confusion_matrix.png', dpi=300)
    plt.close()
    print("Confusion matrix plot saved.")

    print('Analysis complete. Results saved.')


if __name__ == '__main__':
    main()
