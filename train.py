from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, confusion_matrix, recall_score, f1_score
from keras.models import load_model
import numpy as np
from csi import build_csi
from constants import *
from utils import *


def sigmoid_array(x):
    return 1 / (1 + np.exp(-x))


def train_test_fn(eid_train, eid_test, task, eid_list, burnin, X_dict, y_dict, dict_, subX_dict, scaler_dict, model):
    noerr_eid_list = set()

    ### Training... ###
    # acc = 0
    nb_epoch = 30
    if task == "regression":
        eid_train = eid_list
        eid_test = []

    for ep in range(nb_epoch + 1):
        print("{} epoch!!!!!!!!".format(ep))
        ##### Looping for eid_train #####
        losses = []
        for ii, eid in enumerate(eid_train):
            X = X_dict[eid]
            if X.shape[0] <= 2 * burnin:  # ignore length<=1 sequence
                continue
            X = X.astype(np.float32)
            y = y_dict[eid]

            label = int(dict_[eid]['label'])
            if task == "classification":
                assert (label == y)

            noerr_eid_list.add(eid)

            sh = scaler_dict[eid][0]
            si = scaler_dict[eid][1]

            ##### Main input #####
            trainX = X
            ##### Sub input #####
            sub_trainX = subX_dict[eid]

            if task == "regression":
                ### TODO : if we want to predict more features, add here.
                if y.shape[1] > 1:
                    trainY = np.hstack([sh.transform(y[:, 0].reshape(-1, 1)),
                                        si.transform(y[:, 1].reshape(-1, 1))])
                else:
                    trainY = si.transform(y)
                dim_output = trainY.shape

            elif task == "classification":
                trainY = y
                dim_output = 1

            if ep % 50 == 0 and ii % 1000 == 0:
                h = model.fit([trainX[np.newaxis, :, :], sub_trainX[np.newaxis, :, :]], np.array([trainY]),
                              batch_size=1, nb_epoch=1, verbose=2)
            else:
                h = model.fit([trainX[np.newaxis, :, :], sub_trainX[np.newaxis, :, :]], np.array([trainY]),
                              batch_size=1, nb_epoch=1, verbose=0)
            losses.append(h.history['loss'][0])
        print("%% mean loss : {}".format(np.mean(losses)))

        ### Evaluation ###
        preds = []
        rmses = []
        y_test = []
        for ii, eid in enumerate(eid_test):
            X = X_dict[eid]
            if X.shape[0] <= 2 * burnin:  # ignore length<=1 sequence
                continue

            X = X.astype(np.float32)
            y = y_dict[eid]

            testX = X
            sub_testX = subX_dict[eid]

            if task == "classification":
                y_test.append(int(dict_[eid]['label']))

                pred = model.predict([np.array([testX]), np.array([sub_testX])], verbose=0)
                preds.append(pred[0, 0])

            elif task == "regression":
                predict_y = model.predict(np.array([testX]), verbose=0)

                sh = scaler_dict[eid][0]
                si = scaler_dict[eid][1]

                if predict_y.shape[2] == 1:
                    predict_y = np.hstack([sh.inverse_transform(predict_y[0, burnin:, 0].reshape(-1, 1))])
                elif predict_y.shape[2] == 2:
                    predict_y = np.hstack([sh.inverse_transform(predict_y[0, burnin:, 0].reshape(-1, 1)),
                                           si.inverse_transform(predict_y[0, burnin:, 1].reshape(-1, 1))])
                elif predict_y.shape[2] > 2:
                    predict_y = np.hstack([sh.inverse_transform(predict_y[0, burnin:, 0].reshape(-1, 1)),
                                           si.inverse_transform(predict_y[0, burnin:, 1].reshape(-1, 1)),
                                           predict_y[0, burnin:, 2:]])
                nb_features = predict_y.shape[1]
                rmse = np.sqrt(np.mean((predict_y[:-1, :] - trainX[burnin + 1:, :nb_features]) ** 2))
                rmses.append(rmse)

        if task == "classification":
            preds = np.array(preds)
            preds = preds > 0.5
            accuracy = accuracy_score(y_test, preds)
            precision = precision_score(y_test, preds)
            recall = recall_score(y_test, preds)
            f1 = f1_score(y_test, preds)
            tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()

            print("%%% Test results {} samples %%%".format(len(y_test)))
            print("accuracy: {}".format(accuracy))
            print("precision : {:.4f}".format(precision))
            print("recall : {:.4f}".format(recall))
            print("F1 score : {:.4f}".format(f1))

        elif task == "regression":
            print("%%% Test results {} samples".format(len(rmses)))
            print("mean rmse : {}".format(np.mean(rmses)))


if __name__ == "__main__":
    eid_list = load_from_pickle(EID_LIST_PATH)
    X_dict = load_from_pickle(X_DICT_PATH)
    y_dict = load_from_pickle(Y_DICT_PATH)
    dict_ = load_from_pickle(DICT_PATH)
    subX_dict = load_from_pickle(SUBX_DICT_PATH)
    scaler_dict = load_from_pickle(SCALER_DICT_PATH)
    params = load_from_pickle(PARAM_PATH)
    user2ind = load_from_pickle(USER2IND_PATH)
    eid2ind = load_from_pickle(EID2IND_PATH)
    eid_train = load_from_pickle(EID_TRAIN_PATH)
    eid_test = load_from_pickle(EID_TEST_PATH)
    task, nb_feature_sub, burnin = params["task"], params["nb_feature_sub"], params["burnin"]

    model = build_csi(user2ind, eid2ind, nb_feature_sub, task)
    train_test_fn(eid_train, eid_test, task, eid_list, burnin, X_dict, y_dict, dict_, subX_dict, scaler_dict, model)
