from config.environ import random_seed
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import (RandomForestRegressor, AdaBoostRegressor, ExtraTreesRegressor, BaggingRegressor,
    RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier, BaggingClassifier)
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.inspection import permutation_importance
from sklearn.svm import LinearSVR, LinearSVC
from sklearn.linear_model import  LinearRegression, LogisticRegression, RidgeClassifier, Ridge
from sklearn.metrics import mean_squared_error, accuracy_score
from mlens.ensemble import SuperLearner
import numpy as np
import xgboost as xgb
import sys


def DTREE(params, predictor, data_dict):
    if params[predictor]['TEST_VAR'] != "acc":
        tree = DecisionTreeRegressor(max_depth=params[predictor]["MAX_DEPTH"],
            min_samples_leaf=params[predictor]["MIN_SAMP_LEAF"], random_state=random_seed)
    else:
        tree = DecisionTreeClassifier(max_depth=params[predictor]["MAX_DEPTH"],
            min_samples_leaf=params[predictor]["MIN_SAMP_LEAF"], random_state=random_seed)
    tree.fit(data_dict[predictor]["X_train"], data_dict[predictor]["y_train"])
    # imps = permutation_importance(tree, data_dict[predictor]["X_train"],
    #     data_dict[predictor]["y_train"])["importances_mean"]
    # for i,feature in enumerate(params[predictor]["FEATURE_COLUMNS"]):
    #     print(f"{feature} has importance of {imps[i]}")
    # print(data_dict[predictor]['X_test'], data_dict[predictor]['X_test'].shape)
    predicted_value = tree.predict(data_dict[predictor]["X_test"])
    if params[predictor]['TEST_VAR'] != "acc":
        predicted_value = rescale_2D_preds(predictor, data_dict, predicted_value)
    else:
        predicted_value = predicted_value[-1]
    
    return predicted_value

def XTREE(params, predictor, data_dict):
    if params[predictor]['TEST_VAR'] != "acc":
        xtree = ExtraTreesRegressor(n_estimators=params[predictor]["N_ESTIMATORS"],
            max_depth=params[predictor]["MAX_DEPTH"], min_samples_leaf=params[predictor]["MIN_SAMP_LEAF"], 
            random_state=random_seed, n_jobs=-1)
    else:
         xtree = ExtraTreesClassifier(n_estimators=params[predictor]["N_ESTIMATORS"],
            max_depth=params[predictor]["MAX_DEPTH"], min_samples_leaf=params[predictor]["MIN_SAMP_LEAF"], 
            random_state=random_seed, n_jobs=-1)
    xtree.fit(data_dict[predictor]["X_train"], data_dict[predictor]["y_train"])
    predicted_value = xtree.predict(data_dict[predictor]["X_test"])
    if params[predictor]['TEST_VAR'] != "acc":
        predicted_value = rescale_2D_preds(predictor, data_dict, predicted_value)
    else:
        predicted_value = predicted_value[-1]
    
    return predicted_value

def RFORE(params, predictor, data_dict):
    if params[predictor]['TEST_VAR'] != "acc":
        fore = RandomForestRegressor(n_estimators=params[predictor]["N_ESTIMATORS"],
            max_depth=params[predictor]["MAX_DEPTH"], min_samples_leaf=params[predictor]["MIN_SAMP_LEAF"], 
            random_state=random_seed, n_jobs=-1)
    else:
        fore = RandomForestClassifier(n_estimators=params[predictor]["N_ESTIMATORS"],
            max_depth=params[predictor]["MAX_DEPTH"], min_samples_leaf=params[predictor]["MIN_SAMP_LEAF"], 
            random_state=random_seed, n_jobs=-1)
    fore.fit(data_dict[predictor]["X_train"], data_dict[predictor]["y_train"])
    predicted_value = fore.predict(data_dict[predictor]["X_test"])
    if params[predictor]['TEST_VAR'] != "acc":
        predicted_value = rescale_2D_preds(predictor, data_dict, predicted_value)
    else:
        predicted_value = predicted_value[-1]
    
    return predicted_value



def KNN(params, predictor, data_dict):
    if params[predictor]['TEST_VAR'] != "acc":
        knn = KNeighborsRegressor(n_neighbors=params[predictor]["N_NEIGHBORS"], 
            weights=params[predictor]['WEIGHTS'], n_jobs=-1)
    else:
        knn = KNeighborsClassifier(n_neighbors=params[predictor]["N_NEIGHBORS"], 
            weights=params[predictor]['WEIGHTS'], n_jobs=-1)
    knn.fit(data_dict[predictor]["X_train"], data_dict[predictor]["y_train"])
    predicted_value = knn.predict(data_dict[predictor]["X_test"])
    if params[predictor]['TEST_VAR'] != "acc":
        predicted_value = rescale_2D_preds(predictor, data_dict, predicted_value)
    else:
        predicted_value = predicted_value[-1]
    
    return predicted_value


def ADA(params, predictor, data_dict):
    if params[predictor]['TEST_VAR'] != "acc":
        base = DecisionTreeRegressor(max_depth=params[predictor]["MAX_DEPTH"],
            min_samples_leaf=params[predictor]["MIN_SAMP_LEAF"], random_state=random_seed)
        ada = AdaBoostRegressor(base_estimator=base, n_estimators=params[predictor]["N_ESTIMATORS"],
            random_state=random_seed)
    else:
        base = DecisionTreeClassifier(max_depth=params[predictor]["MAX_DEPTH"],
            min_samples_leaf=params[predictor]["MIN_SAMP_LEAF"], random_state=random_seed)
        ada = AdaBoostClassifier(base_estimator=base, n_estimators=params[predictor]["N_ESTIMATORS"],
            random_state=random_seed)
    ada.fit(data_dict[predictor]["X_train"], data_dict[predictor]["y_train"])
    # imps = permutation_importance(ada, data_dict[predictor]["X_train"],
    #     data_dict[predictor]["y_train"])["importances_mean"]
    # for i,feature in enumerate(params[predictor]["FEATURE_COLUMNS"]):
    #     print(f"{feature} has importance of {imps[i]}")
    predicted_value = ada.predict(data_dict[predictor]["X_test"])
    if params[predictor]['TEST_VAR'] != "acc":
        predicted_value = rescale_2D_preds(predictor, data_dict, predicted_value)
    else:
        predicted_value = predicted_value[-1]
    
    return predicted_value


def XGB(params, predictor, data_dict):
    if params[predictor]['TEST_VAR'] != "acc":
        regressor = xgb.XGBRegressor(n_estimators=params[predictor]["N_ESTIMATORS"], 
            max_depth=params[predictor]["MAX_DEPTH"], max_leaves=params[predictor]["MAX_LEAVES"], 
            learning_rate=.05, n_jobs=8, random_state=random_seed, predictor="cpu_predictor")
    else:
        regressor = xgb.XGBClassifier(n_estimators=params[predictor]["N_ESTIMATORS"], 
            max_depth=params[predictor]["MAX_DEPTH"], max_leaves=params[predictor]["MAX_LEAVES"], 
            learning_rate=.05, n_jobs=8, random_state=random_seed, predictor="cpu_predictor",
            eval_metric="auc", use_label_encoder=False)

    regressor.fit(data_dict[predictor]["X_train"], data_dict[predictor]["y_train"], 
    eval_set=[(data_dict[predictor]["X_train"], data_dict[predictor]["y_train"])], verbose=False)
    
    predicted_value = regressor.predict(data_dict[predictor]["X_test"])
    if params[predictor]['TEST_VAR'] != "acc":
        predicted_value = rescale_2D_preds(predictor, data_dict, predicted_value)
    else:
        predicted_value = predicted_value[-1]
    
    return predicted_value

def BAGREG(params, predictor, data_dict):
    if params[predictor]['TEST_VAR'] != "acc":
        base = DecisionTreeRegressor(max_depth=params[predictor]["MAX_DEPTH"],
        min_samples_leaf=params[predictor]["MIN_SAMP_LEAF"], random_state=random_seed)
        bag = BaggingRegressor(base_estimator=base, n_estimators=params[predictor]["N_ESTIMATORS"],
            random_state=random_seed)
    else:
        base = DecisionTreeClassifier(max_depth=params[predictor]["MAX_DEPTH"],
        min_samples_leaf=params[predictor]["MIN_SAMP_LEAF"], random_state=random_seed)
        bag = BaggingClassifier(base_estimator=base, n_estimators=params[predictor]["N_ESTIMATORS"],
            random_state=random_seed)
    bag.fit(data_dict[predictor]["X_train"], data_dict[predictor]["y_train"])
    # imps = permutation_importance(ada, data_dict[predictor]["X_train"],
    #     data_dict[predictor]["y_train"])["importances_mean"]
    # for i,feature in enumerate(params[predictor]["FEATURE_COLUMNS"]):
    #     print(f"{feature} has importance of {imps[i]}")
    predicted_value = bag.predict(data_dict[predictor]["X_test"])
    if params[predictor]['TEST_VAR'] != "acc":
        predicted_value = rescale_2D_preds(predictor, data_dict, predicted_value)
    else:
        predicted_value = predicted_value[-1]

    return predicted_value

def MLP(params, predictor, data_dict):
    if params[predictor]['TEST_VAR'] != "acc":
        if params[predictor]['EARLY_STOP'] == False:
            mlp = MLPRegressor(params[predictor]['LAYERS'], shuffle=False, n_iter_no_change=params[predictor]['PATIENCE'],
                verbose=False)
        else:
            mlp = MLPRegressor(params[predictor]['LAYERS'], early_stopping=True, validation_fraction=params[predictor]['VALIDATION_FRACTION'],
                n_iter_no_change=params[predictor]['PATIENCE'], shuffle=False, verbose=False)
    else:
        if params[predictor]['EARLY_STOP'] == False:
            mlp = MLPClassifier(params[predictor]['LAYERS'], shuffle=False, n_iter_no_change=params[predictor]['PATIENCE'],
                verbose=False)
        else:
            mlp = MLPClassifier(params[predictor]['LAYERS'], early_stopping=True, validation_fraction=params[predictor]['VALIDATION_FRACTION'],
                n_iter_no_change=params[predictor]['PATIENCE'], shuffle=False, verbose=False)

    mlp.fit(data_dict[predictor]["X_train"], data_dict[predictor]["y_train"])
    predicted_value = mlp.predict(data_dict[predictor]["X_test"])
    if params[predictor]['TEST_VAR'] != "acc":
        predicted_value = rescale_2D_preds(predictor, data_dict, predicted_value)
    else:
        predicted_value = predicted_value[-1]

    return predicted_value

def MLENS(params, predictor, data_dict):
    if params[predictor]['TEST_VAR'] != "acc":
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
            ensemble.add_meta(Ridge())
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

    else:
        ensemble = SuperLearner(scorer=accuracy_score, random_state=random_seed, n_jobs=-1, verbose=True)

        for layer in params[predictor]['LAYERS']:
            fully_parameterized_models = []

            for model in layer:
                if "DTREE" in model:
                    sub_model = DecisionTreeClassifier(max_depth=params[predictor][model]["MAX_DEPTH"],
                        min_samples_leaf=params[predictor][model]["MIN_SAMP_LEAF"], random_state=random_seed)
                elif "XTREE" in model:
                    sub_model = ExtraTreesClassifier(n_estimators=params[predictor][model]["N_ESTIMATORS"],
                        max_depth=params[predictor][model]["MAX_DEPTH"], min_samples_leaf=params[predictor][model]["MIN_SAMP_LEAF"], 
                        random_state=random_seed, n_jobs=-1)
                elif "RFORE" in model:
                    sub_model = RandomForestClassifier(n_estimators=params[predictor][model]["N_ESTIMATORS"],
                        max_depth=params[predictor][model]["MAX_DEPTH"], min_samples_leaf=params[predictor][model]["MIN_SAMP_LEAF"], 
                        random_state=random_seed, n_jobs=-1)
                elif "KNN" in model:
                    sub_model = KNeighborsClassifier(n_neighbors=params[predictor][model]["N_NEIGHBORS"], 
                        weights=params[predictor][model]['WEIGHTS'], n_jobs=-1)
                elif "ADA" in model:
                    base = DecisionTreeClassifier(max_depth=params[predictor][model]["MAX_DEPTH"],
                        min_samples_leaf=params[predictor][model]["MIN_SAMP_LEAF"], random_state=random_seed)
                    sub_model = AdaBoostClassifier(base_estimator=base, n_estimators=params[predictor][model]["N_ESTIMATORS"],
                        random_state=random_seed)
                elif "XGB" in model:
                    sub_model = xgb.XGBClassifier(n_estimators=params[predictor][model]["N_ESTIMATORS"], 
                        max_depth=params[predictor][model]["MAX_DEPTH"], max_leaves=params[predictor][model]["MAX_LEAVES"], 
                        learning_rate=.05, n_jobs=8, random_state=random_seed, predictor="cpu_predictor")
                elif "BAGREG" in model:
                    base = DecisionTreeClassifier(max_depth=params[predictor][model]["MAX_DEPTH"],
                        min_samples_leaf=params[predictor][model]["MIN_SAMP_LEAF"], random_state=random_seed)
                    sub_model = BaggingClassifier(base_estimator=base, n_estimators=params[predictor][model]["N_ESTIMATORS"],
                        random_state=random_seed)
                elif "MLP" in model:
                    if params[predictor][model]['EARLY_STOP'] == False:
                        sub_model = MLPClassifier(params[predictor][model]['LAYERS'], shuffle=False, n_iter_no_change=params[predictor][model]['PATIENCE'],
                            verbose=False)
                    else:
                        sub_model = MLPClassifier(params[predictor][model]['LAYERS'], early_stopping=True, validation_fraction=params[predictor][model]['VALIDATION_FRACTION'],
                            n_iter_no_change=params[predictor][model]['PATIENCE'], shuffle=False, verbose=False)
                else:
                    print(f"Submodel {model} was not recognized, try again")
                    sys.exit(-1)

                fully_parameterized_models.append(sub_model)

            ensemble.add(fully_parameterized_models)

                
        if params[predictor]['META_EST'] == "reg":
            ensemble.add_meta(LogisticRegression())
        elif params[predictor]['META_EST'] == "SVR":
            ensemble.add_meta(LinearSVC(random_state=random_seed))
        elif params[predictor]['META_EST'] == "ridge":
            ensemble.add_meta(RidgeClassifier())
        elif params[predictor]['META_EST'] == "DTREE":
            ensemble.add_meta(DecisionTreeClassifier(max_depth=5, random_state=random_seed)) 
        elif params[predictor]['META_EST'] == "RFORE":
            ensemble.add_meta(RandomForestClassifier(random_state=random_seed))
        elif params[predictor]['META_EST'] == "KNN":
            ensemble.add_meta(KNeighborsClassifier())
        elif params[predictor]['META_EST'] == "XGB":
            ensemble.add_meta(xgb.XGBClassifier(n_estimators=50, learning_rate=.05, 
                random_state=random_seed, predictor="cpu_predictor"))
        elif params[predictor]['META_EST'] == "MLP":
            ensemble.add_meta(MLPClassifier((10, 10), shuffle=False, n_iter_no_change=5, verbose=False))
        else:
            print(f"MLENS meta estimator {params[predictor]['META_EST']} not recognized, try again")
            sys.exit(-1)

    
    ensemble.fit(data_dict[predictor]["X_train"], data_dict[predictor]["y_train"])
    predicted_value = ensemble.predict(data_dict[predictor]["X_test"])
    if params[predictor]['TEST_VAR'] != "acc":
        predicted_value = rescale_2D_preds(predictor, data_dict, predicted_value)
    else:
        predicted_value = predicted_value[-1]

    return predicted_value

def rescale_2D_preds(predictor, data_dict, unscaled):
    scale = data_dict[predictor]["column_scaler"]["future"]
    pred = np.array(unscaled)
    pred = pred.reshape(1, -1)
    predicted_value = np.float32(scale.inverse_transform(pred)[-1][-1])

    return predicted_value

