from config.environ import random_seed
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, ExtraTreesRegressor, BaggingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.inspection import permutation_importance
from sklearn.svm import LinearSVR
from sklearn.linear_model import HuberRegressor, LinearRegression
from sklearn.metrics import mean_squared_error
from mlens.ensemble import SuperLearner
from statistics import mean
import numpy as np
import xgboost as xgb
import sys


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

def XTREE(params, predictor, data_dict):
    xtree = ExtraTreesRegressor(n_estimators=params[predictor]["N_ESTIMATORS"],
        max_depth=params[predictor]["MAX_DEPTH"], min_samples_leaf=params[predictor]["MIN_SAMP_LEAF"], 
        random_state=random_seed, n_jobs=-1)
    xtree.fit(data_dict[predictor]["X_train"], data_dict[predictor]["y_train"])
    xtree_pred = xtree.predict(data_dict[predictor]["X_test"])
    predicted_price = rescale_2D_preds(predictor, data_dict, xtree_pred)
    
    return predicted_price

def RFORE(params, predictor, data_dict):
    fore = RandomForestRegressor(n_estimators=params[predictor]["N_ESTIMATORS"],
        max_depth=params[predictor]["MAX_DEPTH"], min_samples_leaf=params[predictor]["MIN_SAMP_LEAF"], 
        random_state=random_seed, n_jobs=-1)
    fore.fit(data_dict[predictor]["X_train"], data_dict[predictor]["y_train"])
    fore_pred = fore.predict(data_dict[predictor]["X_test"])
    predicted_price = rescale_2D_preds(predictor, data_dict, fore_pred)
    
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

def MLP(params, predictor, data_dict):
    if params[predictor]['EARLY_STOP'] == False:
        mlp = MLPRegressor(params[predictor]['LAYERS'], shuffle=False, n_iter_no_change=params[predictor]['PATIENCE'],
            verbose=False)
    else:
        mlp = MLPRegressor(params[predictor]['LAYERS'], early_stopping=True, validation_fraction=params[predictor]['VALIDATION_FRACTION'],
            n_iter_no_change=params[predictor]['PATIENCE'], shuffle=False, verbose=False)
    mlp.fit(data_dict[predictor]["X_train"], data_dict[predictor]["y_train"])
    mlp_pred = mlp.predict(data_dict[predictor]["X_test"])
    predicted_price = rescale_2D_preds(predictor, data_dict, mlp_pred)

    return predicted_price

def MLENS(params, predictor, data_dict):
    ensemble = SuperLearner(scorer=mean_squared_error, random_state=random_seed, n_jobs=-1, verbose=True)

    for layer in params[predictor]['LAYERS']:
        fully_parameterized_models = []

        for model in layer:
            if "DTREE" in model:
                sub_model = DecisionTreeRegressor(max_depth=params[predictor][model]["MAX_DEPTH"],
                    min_samples_leaf=params[predictor][model]["MIN_SAMP_LEAF"], random_state=random_seed)
            elif "XTREE" in model:
                sub_model = ExtraTreesRegressor(n_estimators=params[predictor][model]["N_ESTIMATORS"],
                    max_depth=params[predictor][model]["MAX_DEPTH"], min_samples_leaf=params[predictor][model]["MIN_SAMP_LEAF"], 
                    random_state=random_seed, n_jobs=-1)
            elif "RFORE" in model:
                sub_model = RandomForestRegressor(n_estimators=params[predictor][model]["N_ESTIMATORS"],
                    max_depth=params[predictor][model]["MAX_DEPTH"], min_samples_leaf=params[predictor][model]["MIN_SAMP_LEAF"], 
                    random_state=random_seed, n_jobs=-1)
            elif "KNN" in model:
                sub_model = KNeighborsRegressor(n_neighbors=params[predictor][model]["N_NEIGHBORS"], 
                    weights=params[predictor][model]['WEIGHTS'], n_jobs=-1)
            elif "ADA" in model:
                base = DecisionTreeRegressor(max_depth=params[predictor][model]["MAX_DEPTH"],
                    min_samples_leaf=params[predictor][model]["MIN_SAMP_LEAF"], random_state=random_seed)
                sub_model = AdaBoostRegressor(base_estimator=base, n_estimators=params[predictor][model]["N_ESTIMATORS"],
                    random_state=random_seed)
            elif "XGB" in model:
                sub_model = xgb.XGBRegressor(n_estimators=params[predictor][model]["N_ESTIMATORS"], 
                    max_depth=params[predictor][model]["MAX_DEPTH"], max_leaves=params[predictor][model]["MAX_LEAVES"], 
                    learning_rate=.05, n_jobs=8, random_state=random_seed, predictor="cpu_predictor")
            elif "BAGREG" in model:
                base = DecisionTreeRegressor(max_depth=params[predictor][model]["MAX_DEPTH"],
                    min_samples_leaf=params[predictor][model]["MIN_SAMP_LEAF"], random_state=random_seed)
                sub_model = BaggingRegressor(base_estimator=base, n_estimators=params[predictor][model]["N_ESTIMATORS"],
                    random_state=random_seed)
            elif "MLP" in model:
                if params[predictor][model]['EARLY_STOP'] == False:
                    sub_model = MLPRegressor(params[predictor][model]['LAYERS'], shuffle=False, n_iter_no_change=params[predictor][model]['PATIENCE'],
                        verbose=False)
                else:
                    sub_model = MLPRegressor(params[predictor][model]['LAYERS'], early_stopping=True, validation_fraction=params[predictor][model]['VALIDATION_FRACTION'],
                        n_iter_no_change=params[predictor][model]['PATIENCE'], shuffle=False, verbose=False)
            else:
                print(f"Submodel {model} was not recognized, try again")
                sys.exit(-1)

            fully_parameterized_models.append(sub_model)

        ensemble.add(fully_parameterized_models)

            
    if params[predictor]['META_EST'] == "lin_reg":
        ensemble.add_meta(LinearRegression())
    elif params[predictor]['META_EST'] == "SVR":
        ensemble.add_meta(LinearSVR(random_state=random_seed))
    elif params[predictor]['META_EST'] == "huber":
        ensemble.add_meta(HuberRegressor())
    elif params[predictor]['META_EST'] == "DTREE":
        ensemble.add_meta(DecisionTreeRegressor(max_depth=5, random_state=random_seed)) 
    elif params[predictor]['META_EST'] == "RFORE":
        ensemble.add_meta(RandomForestRegressor(random_state=random_seed))
    elif params[predictor]['META_EST'] == "KNN":
        ensemble.add_meta(KNeighborsRegressor())
    elif params[predictor]['META_EST'] == "XGB":
        ensemble.add_meta(xgb.XGBRegressor(n_estimators=50, learning_rate=.05, 
            random_state=random_seed, predictor="cpu_predictor"))
    elif params[predictor]['META_EST'] == "MLP":
        ensemble.add_meta(MLPRegressor((10, 10), shuffle=False, n_iter_no_change=5, verbose=False))
    else:
        print(f"MLENS meta estimator {params[predictor]['META_EST']} not recognized, try again")
        sys.exit(-1)

    ensemble.fit(data_dict[predictor]["X_train"], data_dict[predictor]["y_train"])
    mlen_pred = ensemble.predict(data_dict[predictor]["X_test"])
    predicted_price = rescale_2D_preds(predictor, data_dict, mlen_pred)
    # scale = data_dict[predictor]["column_scaler"]["future"]
    # mlen_pred = np.array(mlen_pred)
    # mlen_pred = mlen_pred.reshape(1, -1)
    # predicted_price = np.float32(scale.inverse_transform(mlen_pred)[-1][-1])

    return predicted_price

def rescale_2D_preds(predictor, data_dict, unscaled):
    scale = data_dict[predictor]["column_scaler"]["future"]
    pred = np.array(unscaled)
    pred = pred.reshape(1, -1)
    predicted_price = np.float32(scale.inverse_transform(pred)[-1][-1])

    return predicted_price

