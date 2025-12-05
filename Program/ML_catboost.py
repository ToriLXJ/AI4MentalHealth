import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from catboost import CatBoostRegressor

def make_strata(y, q=10):
    try:
        return pd.qcut(y, q=q, duplicates="drop")
    except Exception:
        return None

def train_catboost_model(X, y, selected_vars, random_state=42, test_size=0.3, val_size=0.15):
    """
    训练CatBoost模型
    """
    
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=random_state, 
        stratify=make_strata(y, q=10)
    )
    
    cat_vars = [c for c in selected_vars if c in X.columns and str(X[c].dtype) == "category"]
    num_vars = [c for c in selected_vars if c in X.columns and c not in cat_vars]
    
    # 预处理
    # CatBoost虽然可以原生处理类别，有时候不OneHot直接传类别索引效果更好，但这里为了兼容统一的Pipeline接口先用OHE
    preprocess = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), num_vars),
            ("cat", Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            ]), cat_vars),
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )
    
    # 基础模型
    base_model = CatBoostRegressor(
        loss_function='RMSE',
        random_seed=random_state,
        verbose=False,
        allow_writing_files=False
    )
    
    # CatBoost 搜索空间,针对小样本进行了调整
    param_dist = {
        'cat__iterations': [500, 800, 1000],
        'cat__depth': [4, 6, 8], # CatBoost 默认深度6通常很好
        'cat__learning_rate': [0.01, 0.03, 0.05],
        'cat__l2_leaf_reg': [1, 3, 5, 7], # L2正则化
        'cat__subsample': [0.7, 0.8, 0.9],
        'cat__rsm': [0.7, 0.8, 0.9], # Random Subspace Method (ColSample)
        'cat__min_data_in_leaf': [1, 5, 10, 20]
    }
    
    pipe_cat = Pipeline([
        ("prep", preprocess),
        ("cat", base_model)
    ])
    
    # 随机搜索
    cv = KFold(n_splits=5, shuffle=True, random_state=random_state)
    rs_cat = RandomizedSearchCV(
        estimator=pipe_cat,
        param_distributions=param_dist,
        n_iter=20, # 快速搜索
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        cv=cv,
        refit=True,
        random_state=random_state,
        verbose=1
    )
    
    print("Start CatBoost hyperparameter search...")
    rs_cat.fit(X_tr[selected_vars], y_tr)
    
    print("\n=== CatBoost RandomizedSearchCV ===")
    print("Best CV Score:", rs_cat.best_score_)
    print("Best Params:", rs_cat.best_params_)
    
    # 最终训练（带早停）
    X_tr2, X_val, y_tr2, y_val = train_test_split(
        X_tr[selected_vars], y_tr, test_size=val_size, random_state=random_state,
        stratify=make_strata(y_tr, q=10)
    )
    
    final_preprocess = rs_cat.best_estimator_.named_steps["prep"]
    final_preprocess.fit(X_tr2, y_tr2)
    
    X_tr2_t = final_preprocess.transform(X_tr2)
    X_val_t = final_preprocess.transform(X_val)
    X_te_t = final_preprocess.transform(X_te[selected_vars])
    
    # 提取最佳参数
    best_params = {k.replace("cat__", ""): v for k, v in rs_cat.best_params_.items()}
    
    final_model = CatBoostRegressor(
        loss_function='RMSE',
        random_seed=random_state,
        verbose=100,
        allow_writing_files=False,
        **best_params
    )
    
    # 增加迭代次数供早停使用
    if 'iterations' in best_params:
        final_model.set_params(iterations=2000)
    
    print("\nTraining final CatBoost model...")
    final_model.fit(
        X_tr2_t, y_tr2,
        eval_set=(X_val_t, y_val),
        early_stopping_rounds=50,
        use_best_model=True
    )
    
    # 预测
    pred_tr = final_model.predict(X_tr2_t)
    pred_val = final_model.predict(X_val_t)
    pred_te = final_model.predict(X_te_t)
    
    return {
        'model': final_model,
        'preprocessor': final_preprocess,
        'feature_names': final_preprocess.get_feature_names_out(),
        'X_train': X_tr2, 'X_val': X_val, 'X_test': X_te,
        'y_train': y_tr2, 'y_val': y_val, 'y_test': y_te,
        'pred_val': pred_val, # 返回预测值用于融合
        'pred_test': pred_te, # 返回预测值用于融合
        'train_r2': r2_score(y_tr2, pred_tr),
        'train_rmse': np.sqrt(mean_squared_error(y_tr2, pred_tr)),
        'train_mae': mean_absolute_error(y_tr2, pred_tr),
        'val_r2': r2_score(y_val, pred_val),
        'val_rmse': np.sqrt(mean_squared_error(y_val, pred_val)),
        'val_mae': mean_absolute_error(y_val, pred_val),
        'test_r2': r2_score(y_te, pred_te),
        'test_rmse': np.sqrt(mean_squared_error(y_te, pred_te)),
        'test_mae': mean_absolute_error(y_te, pred_te)
    }