import argparse
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, confusion_matrix, recall_score, f1_score
from keras.models import load_model
import numpy as np
from csi import build_csi, TEXT_FEATURE_DIM
from constants import *
from utils import *
from evaluator import Evaluator
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

METRICS = ["accuracy", "precision", "recall", "f1"]
N_EPOCHS = 200


def get_exp_name(task_name, model_name):
    return "{}-{}-{}".format(task_name, model_name, datetime.now().strftime("%D-%H-%M-%S").replace("/", "_"))


def sigmoid_array(x):
    return 1 / (1 + np.exp(-x))


def train_test_fn(eid_train, eid_val, task, eid_list, burnin, X_dict, y_dict, dict_, subX_dict,
                  X_news_dict, scaler_dict, model, exp_name):

    # for evaluation
    best_evaluator, best_output_path = Evaluator(predictions=[], labels=[]), ""
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

            if score_inputs.shape[2] != 150:
                # pad zeros to news without engagements
                padding = np.zeros(shape=(score_inputs.shape[0], score_inputs.shape[1], TEXT_FEATURE_DIM))
                score_inputs = np.concatenate([score_inputs, padding], axis=-1)

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
        X_news = x_news_dict[eid]
        X = x_dict[remove_tag(eid)]
        if X.shape[0] <= 2 * burnin:  # ignore length<=1 sequence
            continue

        X = X.astype(np.float32)

        testX = X
        sub_testX = sub_x_dict[remove_tag(eid)]
        news_testX = X_news

        y_test.append(int(dict_[remove_tag(eid)]['label']))

        score_inputs = np.array([sub_testX])

        if score_inputs.shape[2] != 150:
            # pad zeros to news without engagements
            padding = np.zeros(shape=(score_inputs.shape[0], score_inputs.shape[1], TEXT_FEATURE_DIM))
            score_inputs = np.concatenate([score_inputs, padding], axis=-1)

        pred = model_obj.predict([np.array([testX]), score_inputs,
                              np.array([news_testX])], verbose=0)
        preds.append(pred[0, 0])

    preds = np.array(preds)
    preds = preds > 0.5
    val_evaluator = Evaluator(y_test, preds)
    validate_result = val_evaluator.evaluate(METRICS)
    writer.add_scalars(tag_name, validate_result, step)
    return val_evaluator


def load_data(use_temporal, use_stance, train_percentage):
    temp_str = "temp" if use_temporal else "no_temp"
    csi_root = "data/csi_{}_{}".format(train_percentage, temp_str)
    eid_list = load_from_pickle(os.path.join(csi_root, "eid_list.pickle"))
    x_dict = load_from_pickle(os.path.join(csi_root, "x_stance_dict.pickle")) if use_stance \
        else load_from_pickle(os.path.join(csi_root, "x_text_dict.pickle"))
    y_dict = load_from_pickle(os.path.join(csi_root, "y_dict.pickle"))
    x_news_dict = load_from_pickle(os.path.join(NEWS_GRAPH_ROOT, "news_source_features.pickle"))
    dict_ = load_from_pickle(os.path.join(csi_root, "dict_.pickle"))
    sub_x_dict = load_from_pickle(os.path.join(csi_root, "subX_dict.pickle"))
    scaler_dict = load_from_pickle(os.path.join(csi_root, "scaler_dict.pickle"))
    params = load_from_pickle(os.path.join(csi_root, "params.pickle"))
    user2ind = load_from_pickle(os.path.join(csi_root, "user2ind.pickle"))
    eid2ind = load_from_pickle(os.path.join(csi_root, "eid2ind.pickle"))
    _eid_train = load_from_pickle(os.path.join(csi_root, "eid_train.pickle"))
    _eid_val = load_from_pickle(os.path.join(csi_root, "eid_val.pickle"))
    _eid_test = load_from_pickle(os.path.join(csi_root, "eid_test.pickle"))
    task, nb_feature_sub, burnin = params["task"], params["nb_feature_sub"], params["burnin"]
    n_engagement_feature = 2 if use_stance else 100
    return eid_list, x_dict, y_dict, x_news_dict, dict_, sub_x_dict, scaler_dict, \
        user2ind, eid2ind, _eid_train, _eid_val, _eid_test, task, nb_feature_sub, burnin, n_engagement_feature


def parse_args():
    parser = argparse.ArgumentParser(description='CSI')
    parser.add_argument('-t', '--temporal', action="store_true")
    parser.add_argument('-s', '--stance', action="store_true")
    parser.add_argument('--percent', type=int)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    use_temporal = args.temporal
    use_stance = args.stance
    train_percentage = args.percent

    eid_list, x_dict, y_dict, x_news_dict, dict_, sub_x_dict, scaler_dict, \
        user2ind, eid2ind, _eid_train, _eid_val, _eid_test, task, nb_feature_sub, burnin, n_engagement_feature = \
        load_data(use_temporal, use_stance, train_percentage)
    exp_name = get_exp_name("{}_{}_{}".format(
        "temp" if use_temporal else "no_temp",
        "stance" if use_stance else "no_stance",
        train_percentage), "csi")
    model = build_csi(user2ind, eid2ind, nb_feature_sub, task, n_engagement_feature)
    best_model_path, _writer = train_test_fn(_eid_train, _eid_val, task, eid_list, burnin, x_dict, y_dict, dict_,
                                             sub_x_dict, x_news_dict, scaler_dict, model, exp_name)
    model.load_weights(best_model_path)
    print("Testing best model")
    eval_fn(model, _eid_test, _writer, N_EPOCHS, "test")
    _writer.close()
