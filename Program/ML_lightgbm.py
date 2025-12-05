import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import lightgbm as lgb
from lightgbm import LGBMRegressor

def make_strata(y, q=10):
    try:
        return pd.qcut(y, q=q, duplicates="drop")
    except Exception:
        # 如果y的分布非常偏态导致无法分层，则退化为不分层
        return None

def train_lightgbm_model(X, y, selected_vars, random_state=1234, test_size=0.3, val_size=0.15, eval_metric = "rmse", 
                         scoring="neg_root_mean_squared_error"):
    
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=random_state, 
        stratify=make_strata(y, q=10)
    )
    
    cat_vars = [c for c in selected_vars if c in X.columns and str(X[c].dtype) in ["category", "object"]]
    num_vars = [c for c in selected_vars if c in X.columns and c not in cat_vars]
    
    # 对于数值变量，增加标准化（虽然树模型不强制，但在某些混合正则化下有帮助）
    # 对于分类变量，忽略出现频率低于 2% 的类别，减少噪音特征
    preprocess = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()) 
            ]), num_vars),
            ("cat", Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=0.02))
            ]), cat_vars),
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )
    
    # 基础模型
    base_model = LGBMRegressor(
        random_state=random_state,
        n_jobs=1,
        verbose=-1,
        importance_type='gain' # 使用增益作为特征重要性评估
    )
    
    # 允许模型学得更"深"一点，但通过极强的正则化(L1/L2)和Dart机制来压制过拟合
    param_dist = {
        'lgbm__boosting_type': ['gbdt', 'dart'], # 允许Dart以增强泛化能力
        'lgbm__objective': ['regression', 'huber'],
        'lgbm__max_depth': [3, 4, 5, 7, -1],
        'lgbm__num_leaves': [7, 15, 31, 50], # 控制树的复杂度
        'lgbm__min_child_samples': [15, 20, 30, 45],
        'lgbm__learning_rate': [0.005, 0.01, 0.03, 0.05],
        'lgbm__n_estimators': [500, 800], 
        'lgbm__subsample': [0.6, 0.7, 0.8, 0.9], 
        'lgbm__subsample_freq': [1, 5],
        'lgbm__colsample_bytree': [0.5, 0.6, 0.7, 0.8],
        
        # 正则化参数
        'lgbm__reg_alpha': [0, 0.1, 1, 5, 10],  # L1 Lasso
        'lgbm__reg_lambda': [0.1, 1, 5, 10, 50] # L2 Ridge
    }
    
    pipe_lgbm = Pipeline([
        ("prep", preprocess),
        ("lgbm", base_model)
    ])
    
    # 交叉验证与搜索
    cv = KFold(n_splits=5, shuffle=True, random_state=random_state)
    
    rs_lgbm = RandomizedSearchCV(
        estimator=pipe_lgbm,
        param_distributions=param_dist,
        n_iter=50,
        scoring=scoring,
        n_jobs=-1,
        cv=cv,
        refit=True,
        random_state=random_state,
        verbose=1
    )
    
    print(f"Start optimized search on {len(X_tr)} samples with {len(selected_vars)} features...")
    rs_lgbm.fit(X_tr[selected_vars], y_tr)
    
    print("\n=== RandomizedSearchCV Results ===")
    print(f"Best CV Score (RMSE): {-rs_lgbm.best_score_:.4f}")
    print("Best Params:", rs_lgbm.best_params_)
    

    X_train_final, X_val_final, y_train_final, y_val_final = train_test_split(
        X_tr, y_tr, test_size=val_size, random_state=random_state,
        stratify=make_strata(y_tr, q=10)
    )
    
    best_pipe = rs_lgbm.best_estimator_
    final_preprocess = best_pipe.named_steps["prep"]
    
    best_lgbm_params = {
        k.replace("lgbm__", ""): v 
        for k, v in rs_lgbm.best_params_.items()
        if k != 'lgbm__n_estimators' # 移除estimators，由早停控制
    }
    
    final_preprocess.fit(X_train_final[selected_vars], y_train_final)
    X_tr_t = final_preprocess.transform(X_train_final[selected_vars])
    X_val_t = final_preprocess.transform(X_val_final[selected_vars])
    X_te_t = final_preprocess.transform(X_te[selected_vars])
    
    # 最终模型训练（带早停）
    final_lgbm = LGBMRegressor(
        n_estimators=3000,
        n_jobs=-1,
        verbose=-1,
        random_state=random_state,
        **best_lgbm_params
    )
    
    print(f"\nTraining final model with Early Stopping (Patience=100)...")
    
    # DART 模式下早停机制比较特殊，将 patience 设大一点 (100) 以防止过早停止
    final_lgbm.fit(
        X_tr_t, y_train_final,
        eval_set=[(X_tr_t, y_train_final), (X_val_t, y_val_final)],
        eval_metric=eval_metric,
        callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=True), 
                   lgb.log_evaluation(period=200)]
    )
    
    pred_tr = final_lgbm.predict(X_tr_t)
    pred_val = final_lgbm.predict(X_val_t)
    pred_te = final_lgbm.predict(X_te_t)
    
    feature_names = final_preprocess.get_feature_names_out()

    return {
        'model': final_lgbm,
        'preprocessor': final_preprocess,
        'feature_names': feature_names,
        'best_params': best_lgbm_params,
        'X_train': X_tr_t, 
        'X_val': X_val_t, 
        'X_test': X_te_t, 
        'y_train': y_train_final,
        'y_val': y_val_final, 
        'y_test': y_te,
        'train_r2': r2_score(y_train_final, pred_tr),
        'train_rmse': np.sqrt(mean_squared_error(y_train_final, pred_tr)),
        'train_mae': mean_absolute_error(y_train_final, pred_tr),
        'val_r2': r2_score(y_val_final, pred_val),
        'val_rmse': np.sqrt(mean_squared_error(y_val_final, pred_val)),
        'val_mae': mean_absolute_error(y_val_final, pred_val),
        'test_r2': r2_score(y_te, pred_te),
        'test_rmse': np.sqrt(mean_squared_error(y_te, pred_te)),
        'test_mae': mean_absolute_error(y_te, pred_te)
    }