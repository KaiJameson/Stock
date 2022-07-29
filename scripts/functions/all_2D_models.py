from config.environ import random_seed
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, ExtraTreesRegressor, BaggingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.inspection import permutation_importance
from sklearn.svm import LinearSVR
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from mlens.ensemble import SuperLearner
from statistics import mean
import numpy as np
import xgboost as xgb


def DTREE(params, predictor, data_dict):
    tree = DecisionTreeRegressor(max_depth=params[predictor]["MAX_DEPTH"],
        min_samples_leaf=params[predictor]["MIN_SAMP_LEAF"], random_state=random_seed)
    tree.fit(data_dict[predictor]["X_train"], data_dict[predictor]["y_train"])
    # imps = permutation_importance(tree, data_dict[predictor]["X_train"],
    #     data_dict[predictor]["y_train"])["importances_mean"]
    # for i,feature in enumerate(params[predictor]["FEATURE_COLUMNS"]):
    #     print(f"{feature} has importance of {imps[i]}")
    tree_pred = tree.predict(data_dict[predictor]["X_test"])
    predicted_price = rescale_2D_preds(predictor, data_dict, tree_pred)
    
    return predicted_price


def RFORE(params, predictor, data_dict):
    fore = RandomForestRegressor(n_estimators=params[predictor]["N_ESTIMATORS"],
        max_depth=params[predictor]["MAX_DEPTH"], min_samples_leaf=params[predictor]["MIN_SAMP_LEAF"], 
        random_state=random_seed, n_jobs=-1)
    fore.fit(data_dict[predictor]["X_train"], data_dict[predictor]["y_train"])
    fore_pred = fore.predict(data_dict[predictor]["X_test"])
    predicted_price = rescale_2D_preds(predictor, data_dict, fore_pred)
    
    return predicted_price

def XTREE(params, predictor, data_dict):
    xtree = ExtraTreesRegressor(n_estimators=params[predictor]["N_ESTIMATORS"],
        max_depth=params[predictor]["MAX_DEPTH"], min_samples_leaf=params[predictor]["MIN_SAMP_LEAF"], 
        random_state=random_seed, n_jobs=-1)
    xtree.fit(data_dict[predictor]["X_train"], data_dict[predictor]["y_train"])
    xtree_pred = xtree.predict(data_dict[predictor]["X_test"])
    predicted_price = rescale_2D_preds(predictor, data_dict, xtree_pred)
    
    return predicted_price


def KNN(params, predictor, data_dict):
    knn = KNeighborsRegressor(n_neighbors=params[predictor]["N_NEIGHBORS"], 
        weights=params[predictor]['WEIGHTS'], n_jobs=-1)
    knn.fit(data_dict[predictor]["X_train"], data_dict[predictor]["y_train"])
    knn_pred = knn.predict(data_dict[predictor]["X_test"])
    predicted_price = rescale_2D_preds(predictor, data_dict, knn_pred)
    
    return predicted_price


def ADA(params, predictor, data_dict):
    base = DecisionTreeRegressor(max_depth=params[predictor]["MAX_DEPTH"],
        min_samples_leaf=params[predictor]["MIN_SAMP_LEAF"], random_state=random_seed)
    ada = AdaBoostRegressor(base_estimator=base, n_estimators=params[predictor]["N_ESTIMATORS"],
        random_state=random_seed)
    ada.fit(data_dict[predictor]["X_train"], data_dict[predictor]["y_train"])
    # imps = permutation_importance(ada, data_dict[predictor]["X_train"],
    #     data_dict[predictor]["y_train"])["importances_mean"]
    # for i,feature in enumerate(params[predictor]["FEATURE_COLUMNS"]):
    #     print(f"{feature} has importance of {imps[i]}")
    ada_pred = ada.predict(data_dict[predictor]["X_test"])
    predicted_price = rescale_2D_preds(predictor, data_dict, ada_pred)
    
    return predicted_price


def XGB(params, predictor, data_dict):
    regressor = xgb.XGBRegressor(n_estimators=params[predictor]["N_ESTIMATORS"], 
        max_depth=params[predictor]["MAX_DEPTH"], max_leaves=params[predictor]["MAX_LEAVES"], 
        learning_rate=.05, n_jobs=8, random_state=random_seed, predictor="cpu_predictor")

    regressor.fit(data_dict[predictor]["X_train"], data_dict[predictor]["y_train"], 
    eval_set=[(data_dict[predictor]["X_train"], data_dict[predictor]["y_train"])], verbose=False)
    xgb_pred = regressor.predict(data_dict[predictor]["X_test"])
    predicted_price = rescale_2D_preds(predictor, data_dict, xgb_pred)
    
    return predicted_price

def BAGREG(params, predictor, data_dict):
    base = DecisionTreeRegressor(max_depth=params[predictor]["MAX_DEPTH"],
        min_samples_leaf=params[predictor]["MIN_SAMP_LEAF"], random_state=random_seed)
    bag = BaggingRegressor(base_estimator=base, n_estimators=params[predictor]["N_ESTIMATORS"],
        random_state=random_seed)
    bag.fit(data_dict[predictor]["X_train"], data_dict[predictor]["y_train"])
    # imps = permutation_importance(ada, data_dict[predictor]["X_train"],
    #     data_dict[predictor]["y_train"])["importances_mean"]
    # for i,feature in enumerate(params[predictor]["FEATURE_COLUMNS"]):
    #     print(f"{feature} has importance of {imps[i]}")
    bag_pred = bag.predict(data_dict[predictor]["X_test"])
    predicted_price = rescale_2D_preds(predictor, data_dict, bag_pred)
    
    return predicted_price

def rescale_2D_preds(predictor, data_dict, unscaled):
    scale = data_dict[predictor]["column_scaler"]["future"]
    pred = np.array(unscaled)
    pred = pred.reshape(1, -1)
    predicted_price = np.float32(scale.inverse_transform(pred)[-1][-1])

    return predicted_price

