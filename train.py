from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, confusion_matrix, recall_score, f1_score
from keras.models import load_model
import numpy as np
from csi import build_csi
from constants import *
from utils import *
from evaluator import Evaluator
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

METRICS = ["accuracy", "precision", "recall", "f1"]
N_EPOCHS = 50


def get_exp_name(task_name, model_name):
    return "{}-{}-{}".format(task_name, model_name, datetime.now().strftime("%D-%H-%M-%S").replace("/", "_"))


def sigmoid_array(x):
    return 1 / (1 + np.exp(-x))


def train_test_fn(eid_train, eid_val, task, eid_list, burnin, X_dict, y_dict, dict_, subX_dict,
                  X_news_dict, scaler_dict, model):

    # for evaluation
    best_evaluator, best_output_path = Evaluator(predictions=[], labels=[]), ""
    exp_name = get_exp_name("fang", "csi")
    save_dir = os.path.join("exp_ckpt", exp_name)
    os.makedirs(save_dir)
    best_dir_path = os.path.join(save_dir, "best.ckpt")
    log_dir = os.path.join("exp_log", exp_name)
    os.makedirs(log_dir)
    writer = SummaryWriter(log_dir)

    noerr_eid_list = set()

    ### Training... ###
    # acc = 0

    for ep in range(N_EPOCHS + 1):
        print("{} epoch!!!!!!!!".format(ep))
        ##### Looping for eid_train #####
        losses = []
        for ii, eid in enumerate(eid_train):
            X_news = X_news_dict[eid]
            X = X_dict[remove_tag(eid)]
            if X.shape[0] <= 2 * burnin:  # ignore length<=1 sequence
                continue
            X = X.astype(np.float32)
            y = y_dict[remove_tag(eid)]

            label = int(dict_[remove_tag(eid)]['label'])
            assert (label == y)

            noerr_eid_list.add(eid)

            ##### Main input #####
            trainX = X
            ##### Sub input #####
            sub_trainX = subX_dict[remove_tag(eid)]

            trainY = y

            capture_inputs = trainX[np.newaxis, :, :]
            score_inputs = sub_trainX[np.newaxis, :, :]
            news_inputs = X_news[np.newaxis, :]

            if ep % 50 == 0 and ii % 1000 == 0:
                h = model.fit([capture_inputs, score_inputs, news_inputs], np.array([trainY]),
                              batch_size=1, nb_epoch=1, verbose=2)
            else:
                h = model.fit([capture_inputs, score_inputs, news_inputs], np.array([trainY]),
                              batch_size=1, nb_epoch=1, verbose=0)
            losses.append(h.history['loss'][0])
        print("%% mean loss : {}".format(np.mean(losses)))

        ### Evaluation ###
        val_evaluator = eval_fn(model, eid_val, writer, ep, "validate")
        if val_evaluator.is_better_than(best_evaluator, METRICS):
            print("Best evaluator is updated.")
            best_evaluator = val_evaluator
            model.save_weights(best_dir_path)

    return best_dir_path, writer


def eval_fn(model_obj, eid_val, writer, step, tag_name="validate"):
    preds, y_test = [], []
    for ii, eid in enumerate(eid_val):
        X_news = X_news_dict[eid]
        X = X_dict[remove_tag(eid)]
        if X.shape[0] <= 2 * burnin:  # ignore length<=1 sequence
            continue

        X = X.astype(np.float32)

        testX = X
        sub_testX = subX_dict[remove_tag(eid)]
        news_testX = X_news

        y_test.append(int(dict_[remove_tag(eid)]['label']))

        pred = model_obj.predict([np.array([testX]), np.array([sub_testX]),
                              np.array([news_testX])], verbose=0)
        preds.append(pred[0, 0])

    preds = np.array(preds)
    preds = preds > 0.5
    val_evaluator = Evaluator(y_test, preds)
    validate_result = val_evaluator.evaluate(METRICS)
    writer.add_scalars(tag_name, validate_result, step)
    return val_evaluator


if __name__ == "__main__":
    eid_list = load_from_pickle(EID_LIST_PATH)
    X_dict = load_from_pickle(X_DICT_PATH)
    y_dict = load_from_pickle(Y_DICT_PATH)
    X_news_dict = load_from_pickle(os.path.join(NEWS_GRAPH_ROOT, "news_source_features.pickle"))
    dict_ = load_from_pickle(DICT_PATH)
    subX_dict = load_from_pickle(SUBX_DICT_PATH)
    scaler_dict = load_from_pickle(SCALER_DICT_PATH)
    params = load_from_pickle(PARAM_PATH)
    user2ind = load_from_pickle(USER2IND_PATH)
    eid2ind = load_from_pickle(EID2IND_PATH)
    _eid_train = load_from_pickle(EID_TRAIN_PATH)
    _eid_val = load_from_pickle(EID_TEST_PATH)
    _eid_test = load_from_pickle(EID_TEST_PATH)
    task, nb_feature_sub, burnin = params["task"], params["nb_feature_sub"], params["burnin"]

    model = build_csi(user2ind, eid2ind, nb_feature_sub, task)
    best_model_path, _writer = train_test_fn(_eid_train, _eid_val, task, eid_list, burnin, X_dict, y_dict, dict_, subX_dict, X_news_dict,
                                             scaler_dict, model)
    model.load_weights(best_model_path)
    eval_fn(model, _eid_test, _writer, N_EPOCHS, "test")
    _writer.close()
