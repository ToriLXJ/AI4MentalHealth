import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb
from xgboost import XGBRegressor, XGBClassifier

def make_strata(y, q=10):
    try:
        return pd.qcut(y, q=q, duplicates="drop")
    except Exception:
        return None

def train_xgboost_model(X, y, selected_vars, random_state=42, test_size=0.3, val_size=0.15):
    # 划分训练集和测试集
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=random_state, 
        stratify=make_strata(y, q=10)
    )
    
    # 变量类型定义
    BINARY_VARS = [
        "gender_T1", "residence", "media_interact_T1", "ace", 
        "IPAQ_T1_1_bin", "IPAQ_T1_3_bin", "IPAQ_T1_5_bin", "marrige_par_bin"
    ]
    
    # 识别分类变量和数值变量
    cat_vars = [c for c in selected_vars if c in X.columns and str(X[c].dtype) == "category"]
    num_vars = [c for c in selected_vars if c in X.columns and c not in cat_vars]
    
    # 数据预处理管道
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
    
    # 根据模型类型选择基础模型
    base_model = XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        random_state=random_state,
        n_estimators=600,
        n_jobs=-1,
        eval_metric="rmse"
    )
    scoring = "neg_root_mean_squared_error"
    
    # 创建管道
    pipe_xgb = Pipeline([
        ("prep", preprocess),
        ("xgb", base_model)
    ])
    
    # 超参数搜索空间
    param_dist = {
        "xgb__max_depth": [2, 3, 4, 5],
        "xgb__min_child_weight": [5, 6, 8, 10, 15, 20],
        "xgb__reg_alpha": [0.3, 0.5, 1, 2, 3, 5, 7],
        "xgb__reg_lambda": [1, 2, 3, 5, 7, 10, 15],
        "xgb__subsample": [0.5, 0.6, 0.7, 0.8, 0.9],
        "xgb__colsample_bytree": [0.4, 0.5, 0.6, 0.7, 0.8],
        "xgb__colsample_bylevel": [0.4, 0.5, 0.6, 0.7, 0.8],
        "xgb__learning_rate": [0.01, 0.02, 0.03, 0.05, 0.08],
        "xgb__gamma": [0.1, 0.3, 0.5, 1.0, 1.5],
        "xgb__n_estimators": [400, 600, 800, 1000, 1500, 2000],
    }
    
    # 交叉验证
    cv = KFold(n_splits=5, shuffle=True, random_state=random_state)
    rs_xgb = RandomizedSearchCV(
        estimator=pipe_xgb,
        param_distributions=param_dist,
        n_iter=100,
        scoring=scoring,
        n_jobs=-1,
        cv=cv,
        refit=True,
        random_state=random_state,
        error_score="raise",
        verbose=2
    )
    
    # 超参数搜索
    print("Start hyperparameter search...")
    rs_xgb.fit(X_tr[selected_vars], y_tr)
    print("\n=== RandomizedSearchCV ===")
    print("Best CV Score:", rs_xgb.best_score_)
    print("Best Params:", rs_xgb.best_params_)
    
    # 获取最佳参数
    best_p = rs_xgb.best_params_.copy()
    n_estimators = best_p.pop("xgb__n_estimators")
    
    # 划分验证集用于早停
    X_tr2, X_val, y_tr2, y_val = train_test_split(
        X_tr[selected_vars], y_tr, test_size=val_size, random_state=random_state,
        stratify=make_strata(y_tr, q=10)
    )
    
    # 拟合预处理器
    final_preprocess = rs_xgb.best_estimator_.named_steps["prep"]
    final_preprocess.fit(X_tr2, y_tr2)
    
    # 转换数据集
    X_tr2_t = final_preprocess.transform(X_tr2)
    X_val_t = final_preprocess.transform(X_val)
    X_te_t = final_preprocess.transform(X_te[selected_vars])
    
    # 准备XGBoost参数
    params = {
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "eval_metric": "rmse",
        "random_state": random_state,
        **{k.replace("xgb__", ""): v for k, v in best_p.items()}
    }
    num_boost_round = n_estimators
    
    # 转换为DMatrix
    dtrain = xgb.DMatrix(X_tr2_t, label=y_tr2)
    dvalid = xgb.DMatrix(X_val_t, label=y_val)
    dtest = xgb.DMatrix(X_te_t)
    
    # 训练模型（带早停）
    watchlist = [(dtrain, "train"), (dvalid, "eval")]
    bst = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=num_boost_round,
        evals=watchlist,
        early_stopping_rounds=30,
        verbose_eval=50
    )
    
    # 预测
    pred_tr = bst.predict(dtrain, iteration_range=(0, bst.best_iteration + 1))
    pred_val = bst.predict(dvalid, iteration_range=(0, bst.best_iteration + 1))
    pred_te = bst.predict(dtest, iteration_range=(0, bst.best_iteration + 1))
    
    # 评估指标
    r2_tr = r2_score(y_tr2, pred_tr)
    rmse_tr = np.sqrt(mean_squared_error(y_tr2, pred_tr))
    mae_tr = mean_absolute_error(y_tr2, pred_tr)
    
    r2_val = r2_score(y_val, pred_val)
    rmse_val = np.sqrt(mean_squared_error(y_val, pred_val))
    mae_val = mean_absolute_error(y_val, pred_val)
    
    r2_te = r2_score(y_te, pred_te)
    rmse_te = np.sqrt(mean_squared_error(y_te, pred_te))
    mae_te = mean_absolute_error(y_te, pred_te)
    
    # 获取特征名称
    feature_names = final_preprocess.get_feature_names_out()

    return {
        'model': bst,
        'preprocessor': final_preprocess,
        'feature_names': feature_names,
        'X_train': X_tr2,
        'X_val': X_val,
        'X_test': X_te,
        'y_train': y_tr2,
        'y_val': y_val, 
        'y_test': y_te,
        'train_r2': r2_tr,
        'train_rmse': rmse_tr,
        'train_mae': mae_tr,
        'val_r2': r2_val,
        'val_rmse': rmse_val,
        'val_mae': mae_val,
        'test_r2': r2_te,
        'test_rmse': rmse_te,
        'test_mae': mae_te
    }