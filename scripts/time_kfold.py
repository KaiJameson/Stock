from sklearn.model_selection import TimeSeriesSplit

from paca_model import nn_train_save

for predictor in testing_list:
    for symbol in something:
        params = {
            # "ENSEMBLE": ["nn1", "nn2"],
            # "ENSEMBLE": ["ADA1", "KNN1", "RFORE1"],
            "ENSEMBLE": ["nn1"],
            "TRADING": False,
            "SAVE_FOLDER": "tune4",
            "nn1" : { 
                "N_STEPS": 5,
                "LOOKUP_STEP": 1,
                "TEST_SIZE": 0.2,
                "LAYERS": [(256, LSTM), (256, Dense), (128, Dense), (64, Dense)],
                "DROPOUT": .4,
                "BIDIRECTIONAL": False,
                "LOSS": "huber_loss",
                "OPTIMIZER": "adam",
                "BATCH_SIZE": 1024,
                "EPOCHS": 2000,
                "PATIENCE": 200,
                "LIMIT": 4000,
                "FEATURE_COLUMNS": ["o", "l", "c"],
                "TEST_VAR": "c",
                "SAVE_PRED": {}
                },
            "LIMIT": 4000,
        }

        symbol = "AGYS"
        predictor = "nn1"
        scale = False
        to_print = True


        df = get_proper_df(symbol, params[predictor]["LIMIT"], "V2")
        data_dict = load_all_data(params, df)


        tt_df, result = preprocess_dfresult(params[predictor], df, scale=scale, to_print=to_print)

        X, y = construct_3D_np(tt_df, params, result)
        print(len(X), len(y))

        num_splits = 5

        accuracies = []
        # kfold = KFold(n_splits=num_splits, shuffle=True)
        kfold = TimeSeriesSplit(n_splits=num_splits)
        i = 0
        for train, test in kfold.split(X, y):
            print(f" IIIIIIIIIIIIIIII {i} \n\n ")
            print(f"len of train {len(X[train])}")
            print(f"len of test {len(y[test])}")
            print(f"what we're selecting {test}")
            i += 1

            train = Dataset.from_tensor_slices((X[train], y[train]))
            valid = Dataset.from_tensor_slices((X[test], y[test]))

            train = train.batch(params[predictor]["BATCH_SIZE"])
            valid = valid.batch(params[predictor]["BATCH_SIZE"])

            train = train.cache()
            valid = valid.cache()

            train = train.prefetch(buffer_size=AUTOTUNE)
            valid = valid.prefetch(buffer_size=AUTOTUNE)

            data_dict["train"] = train
            data_dict["valid"] = valid

            epochs = nn_train_save(symbol, params, predictor=predictor)
            nn_params = params[predictor]
            
            check_model_folders(params["SAVE_FOLDER"], symbol)

            model_name = (symbol + "-" + get_model_name(nn_params))


            model = create_model(nn_params)

            logs_dir = "logs/" + get_time_string() + "-" + params["SAVE_FOLDER"]

            checkpointer = ModelCheckpoint(directory_dict["model"] + "/" + params["SAVE_FOLDER"] + "/" 
                + model_name + ".h5", save_weights_only=True, save_best_only=True, verbose=1)
            
            if save_logs:
                tboard_callback = TensorBoard(log_dir=logs_dir, profile_batch="200, 1200") 
            else:
                tboard_callback = TensorBoard(log_dir=logs_dir, profile_batch=0)

            # early_stop = EarlyStopping(patience=nn_params["PATIENCE"])
            
            history = model.fit(data_dict["train"],
                batch_size=nn_params["BATCH_SIZE"],
                epochs=nn_params["EPOCHS"],
                verbose=0,
                # validation_data=data_dict["valid"],
                callbacks = [tboard_callback, checkpointer]   
            )

            epochs_used = len(history.history["loss"])
                
            if not save_logs:
                delete_files_in_folder(logs_dir)
                os.rmdir(logs_dir)


            print(result["column_scaler"])
            y_real, y_pred = return_real_predict(model, X[test], y[test], result["column_scaler"]["c"])

            # y_real = y[test]
            # y_pred = model.predict(X[test])
            acc = get_accuracy(y_pred, y_real, lookup_step=1)
            print(r1002(acc))
            accuracies.append(acc)
            model.evaluate(valid)


        overall_acc = r1002(sum(accuracies) / num_splits)
        print(overall_acc)