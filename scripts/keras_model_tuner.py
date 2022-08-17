from config.silen_ten import silence_tensorflow
silence_tensorflow()
from tensorflow.keras.layers import LSTM, GRU, Dense, SimpleRNN, Dropout
from config.environ import directory_dict
from config.symbols import sym_dict, keras_tuner_dict
from config.model_repository import keras_tune_models
from functions.data_load import get_proper_df, load_all_data
from functions.time import get_current_datetime
from functions.functions import check_directories, get_model_name, check_prediction_subfolders, r2
from functions.tuner import get_user_input
from paca_model import configure_gpu
import keras_tuner as kt
import tensorflow as tf
import keras





def keras_tuning(params):
    configure_gpu()


    tune_symbols, params = get_user_input(sym_dict, params)
    
    
    
    for symbol in tune_symbols:        
        for predictor in params['ENSEMBLE']:
            save_folder = f"summary-{get_model_name(params[predictor])}"
            check_prediction_subfolders(directory_dict['keras_tuner'], save_folder)

            df = get_proper_df(symbol, 4000, "V2")
            data_dict = load_all_data(params, df, get_current_datetime())

            hyper_params = set()

            def model_builder(hp):
                model = keras.Sequential()
                hp_units = hp.Int("units_l1", min_value=64, max_value=512, step=64)
                model.add(LSTM(units=hp_units, activation='relu', input_shape=(None, len(params[predictor]['FEATURE_COLUMNS'])), return_sequences=False))
                hyper_params.add("units_l1")
                dp_ratio1 = hp.Float("dp_1", min_value=0.0, max_value=.8, step=.2)
                model.add(Dropout(dp_ratio1))
                hyper_params.add("dp_1")
                hp_units2 = hp.Int("units_l2", min_value=64, max_value=512, step=64)
                model.add(Dense(units=hp_units2, activation='relu'))
                hyper_params.add("units_l2")
                dp_ratio2 = hp.Float("dp_2", min_value=0.0, max_value=.8, step=.2)
                model.add(Dropout(dp_ratio2))
                hyper_params.add("dp_2")

                model.add(Dense(1, activation="linear"))
                model.compile(optimizer=keras.optimizers.Adam(),
                                loss=params[predictor]['LOSS'], metrics=["mean_absolute_error"])

                return model


            tuner = kt.Hyperband(model_builder,
                objective="val_loss",
                max_epochs=2000,
                factor=3,
                directory=directory_dict['keras_tuner'],
                project_name=f"{symbol}-{get_model_name(params[predictor])}")

            stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50)



            tuner.search(data_dict[predictor]['train'], validation_data=data_dict[predictor]['valid'], epochs=50, callbacks=[stop_early])
           
           
            best_hyperparams = tuner.get_best_hyperparameters(10)
            print(f"\nThe results from tuning {get_model_name(params[predictor])} on symbol {symbol}:")
            for i, model in enumerate(best_hyperparams):
                print(f"\nResults from model {i + 1} of {len(best_hyperparams)}:")
                for parameter in hyper_params:
                    print(f"Hyperparameter {parameter} had value of {r2(model.get(parameter))}")

            with open(f"{directory_dict['keras_tuner']}/{save_folder}/{symbol}.txt", "a") as f:
                f.write(f"\nThe results from tuning {get_model_name(params[predictor])} on symbol {symbol}:\n")
                for i, model in enumerate(best_hyperparams):
                    f.write(f"\nResults from model {i + 1} of {len(best_hyperparams)}:\n")
                    for parameter in hyper_params:
                        f.write(f"Hyperparameter {parameter} had value of {r2(model.get(parameter))}\n")
                

if __name__ == "__main__":
    check_directories()
    

    for model in keras_tuner_dict["ENSEMBLE"]:
        if model in keras_tune_models:
            keras_tuner_dict[model] = keras_tune_models[model]
    print(keras_tuner_dict)

    keras_tuning(keras_tuner_dict)
