from constants import *
from collections import Counter
import json
import numpy as np
import os
import pickle
import time
from time import mktime
from scipy.sparse import csr_matrix
from sklearn.utils.extmath import randomized_svd
from doc2vec import train_doc2vec, build_doc2vec_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from utils import *
from stance import *


def load_event_tweet_stance_dict():
    event_tweet_stance_dict = {}
    for stance_label in CHOSEN_STANCES:
        stance_path = os.path.join(NEWS_GRAPH_ROOT, "{}.tsv".format(stance_label))
        stance_data = read_csv(stance_path, True, "\t")
        for row in stance_data:
            news, user = remove_tag(row[1]), remove_tag(row[0])
            if news not in event_tweet_stance_dict:
                event_tweet_stance_dict[news] = {}
            event_tweet_stance_dict[news][user] = stance_label
    return event_tweet_stance_dict


def get_user_event_matrix(dict_, u_sample, user2ind, binary=False):
    '''Get (user,event) matrix.
    This matrix will be decomposed by TruncatedSVD (or else?)
    Only users in u_sample are considered.
    '''
    row = []
    col = []
    data = []
    jj = 0
    eid2ind = {}
    for ii, (eid, value) in enumerate(dict_.items()):
        user_in_event = get_user_in_event(dict_, eid, u_sample)
        if len(user_in_event)==0:
            # No user in u_sample appears in this eid event.
            continue
        else:
            eid2ind[eid] = jj
#             eind = eid2ind[eid]
        for uid, nb_occur in user_in_event:
            uind = user2ind[uid]
            col.append(jj)
            row.append(uind)
            if binary:
                data.append(1)    # Binary matrix
            else:
                data.append(nb_occur)
        jj+=1
    print("{} events have at least one user in u_sample".format(jj))
    print("{} events have no user in u_sample".format(len(dict_)-jj))
    return csr_matrix((data, (row, col)), shape=(len(user2ind), len(eid2ind))), eid2ind


def convert_unix_ts(created_at):
    return mktime(time.strptime(created_at, '%a %b %d %H:%M:%S +0000 %Y'))


def get_stats(dict_):
    _nb_messages = []
    _eid_list = []
    _user_set = set()
    _list_lengths = []
    for eid, messages in dict_.items():
        _nb_messages.append(len(messages['timestamps']))
        _eid_list.append(eid)

        _user_set.update(messages['uid'])
        ts = np.array(messages['timestamps'], dtype=np.float32)
        _list_lengths.append(ts[-1] - ts[0])

    return _nb_messages, _eid_list, _user_set, _list_lengths


def load_or_create_dataset(csi_root, temporal=True):
    dataset_output_path = os.path.join(csi_root, "processed_csi.pickle")

    if os.path.exists(dataset_output_path):
        print("Load preprocessed csi dataset at {}".format(dataset_output_path))
        with open(dataset_output_path, "rb") as f:
            return pickle.load(f)

    csi_dir = os.path.join(csi_root, "csi")
    event_tweet_stance_dict = load_event_tweet_stance_dict()
    csi_data = {}

    for label in os.listdir(csi_dir):
        csi_label_dir = os.path.join(csi_dir, label)

        for i, eid in tqdm(enumerate(os.listdir(csi_label_dir)), desc="Loading events"):
            eid_dir = os.path.join(csi_label_dir, eid)
            tweet_dir = os.path.join(eid_dir, "tweets")
            csi_data[eid] = {
                "timestamps": [],
                "uid": [],
                "text": [],
                "label": 1 if label == "real" else 0,
                "stances": []
            }
            engagements = []
            if os.path.exists(tweet_dir):
                for tweet_id in os.listdir(tweet_dir):
                    tweet_path = os.path.join(tweet_dir, tweet_id)
                    with open(tweet_path, "r") as f:
                        tweet_content = json.load(f)
                    unix_ts = convert_unix_ts(tweet_content["created_at"]) if temporal else 0.
                    uid = tweet_content["user"]["id"]
                    text = tweet_content["text"]
                    stance = event_tweet_stance_dict[eid][str(uid)] \
                        if eid in event_tweet_stance_dict and str(uid) in event_tweet_stance_dict[eid] \
                        else "none"
                    engagements.append({
                        "ts": unix_ts,
                        "uid": uid,
                        "text": text,
                        "stance": stance
                    })
                engagements = sorted(engagements, key=lambda k: k["ts"])
                start_ts = engagements[0]["ts"]
                for e in engagements:
                    csi_data[eid]["timestamps"].append(e["ts"] - start_ts)
                    csi_data[eid]["uid"].append(e["uid"])
                    csi_data[eid]["text"].append(e["text"])
                    csi_data[eid]["stances"].append(e["stance"])
            else:
                # add a dummy engagement for news that has no engagement
                csi_data[eid]["timestamps"].append(0.)
                csi_data[eid]["uid"].append(0)
                csi_data[eid]["text"].append("none")
                csi_data[eid]["stances"].append("support")

    with open(dataset_output_path, "wb") as f:
        pickle.dump(csi_data, f)
    return csi_data


def get_Usample(dict_, most_common=50):
    '''Get U_sample who are most_common.'''
    u_sample = []
    cnt = Counter()
    for ii, (eid, value) in enumerate(dict_.items()):
        users = value['uid']
        cnt.update(users)
    return cnt.most_common(most_common)    # [(user_id, #occur in all events), ...]


def get_user_in_event(dict_, eid, u_sample):
    """Get users who acts on a given event, eid"""
    value = dict_[eid]
    cnt = Counter(value['uid'])
    users = set(value['uid'])
    user_in_event = []
    for uid, nb_occur in u_sample:
        if uid in users:
            #  user_in_event.append((uid, nb_occur))
            #  [(user_id, #occur in all events), (user_id, #occur in all events), ...]
            user_in_event.append((uid, cnt[uid]))    # [(user_id, #occur in eid), (user_id, #occur in eid), ...]
    return user_in_event    # [(user_id, #occur in eid), (user_id, #occur in eid), ...]


def get_user_feature_in_event(dict_, eid, u_sample, user_feature_sub, user_sample2ind, user_text_feature=None):
    '''Get user_feature_sub matrix for event eid'''
    user_in_event = get_user_in_event(dict_, eid, u_sample)
    nb_feature = user_feature_sub.shape[1]

    for uid, nb_occur in user_in_event:
        uind = user_sample2ind[uid]
        structure_vec = user_feature_sub[uind, :].reshape(1, -1)
        if user_text_feature is not None:
            text_vec = user_text_feature[uid] if uid in user_text_feature else np.zeros(100)
            text_vec = text_vec.reshape((1, 100))
            feature_vec = np.concatenate([structure_vec, text_vec], axis=1)
        else:
            feature_vec = structure_vec
        try:
            ret_matrix = np.concatenate((ret_matrix, feature_vec), axis=0)
        except:
            ret_matrix = feature_vec
    try:
        return ret_matrix
    except:
        ### if user_in_event is empty
        return np.zeros((1, nb_feature))


def get_doc2vec(doc2vec_model, eid, nonzero_bins):
    for bid, bin_left in enumerate(nonzero_bins):
        if isinstance(eid, int):
            eid_str = str(eid)
        else:
            eid_str = eid
        tag = eid_str+'_'+str(bid)
        temp = doc2vec_model.docvecs[tag]  # (300,)
        temp = temp.reshape(1,-1)
        try:
            X_text = np.concatenate((X_text, temp), axis=0)
        except:
            X_text = temp
    return X_text


def get_stance_mtx(stance_vt_dict, eid, nonzero_bins):
    stance_vectors = []
    for bid, bin_left in enumerate(nonzero_bins):
        event_bin_tag = str(eid) + '_' + str(bid)
        stance_vt = stance_vt_dict[event_bin_tag]  # (300,)
        stance_vectors.append(stance_vt)
    x_stance = np.vstack(stance_vectors) if len(stance_vectors) > 1 else stance_vectors[0].reshape(1, -1)
    return x_stance


def get_features(eid, timestamps, stance_vt_dict, threshold=90, resolution='day',
                 doc2vec_model=None, user_feature=None, user2ind=None, user_list=None,
                 cutoff=50):
    '''
    timestamps
        : relative timestamps since the first tweet
        : it should be sorted.
        : unit = second
    unit of threshold and resolution should be matched.
    '''
    ts = timestamps
    if resolution == 'day':
        binsize = 3600 * 24
    elif resolution == 'hour':
        binsize = 3600
    elif resolution == 'minute':
        binsize = 60
    cnt, bins = np.histogram(ts, bins=range(0, threshold * binsize, binsize))

    nonzero_bins_ind = np.nonzero(cnt)[0]
    nonzero_bins = bins[nonzero_bins_ind]

    hist = cnt[nonzero_bins_ind]
    inv = nonzero_bins_ind[1:] - nonzero_bins_ind[:-1]
    intervals = np.insert(inv, 0, 0)
    ### Cutoff sequence
    #     cutoff = 50
    if len(hist) > cutoff:
        hist = hist[:cutoff]
        intervals = intervals[:cutoff]
        nonzero_bins = nonzero_bins[:cutoff]

    ### user feature
    X_useridx = []
    for bid, bin_left in enumerate(nonzero_bins):
        bin_userlist = []
        bin_right = bin_left + binsize
        try:
            del temp
        except:
            pass
        # Collecting text to make doc
        for tid, t in enumerate(ts):
            if t < bin_left:
                continue
            elif t >= bin_right:
                break
            else:
                pass
            uid = user2ind[user_list[tid]]
            bin_userlist.append(user_list[tid])
            coef = user_feature[uid, :].reshape(1, -1)  # (1,n_components)
            try:
                temp = np.concatenate((temp, coef), axis=0)
            except:
                temp = coef

        X_user_bin = np.mean(temp, axis=0).reshape(1, -1)

        try:
            X_user = np.concatenate((X_user, X_user_bin), axis=0)
        except:
            X_user = X_user_bin
        X_useridx.append(bin_userlist)

    # text feature
    text_matrix = get_doc2vec(doc2vec_model, eid, nonzero_bins)
    text_feature_mtx = np.hstack([hist.reshape(-1, 1), intervals.reshape(-1, 1), X_user, text_matrix])
    stance_matrix = get_stance_mtx(stance_vt_dict, eid, nonzero_bins)
    stance_feature_mtx = np.hstack([hist.reshape(-1, 1), intervals.reshape(-1, 1), X_user, stance_matrix])
    return text_feature_mtx, stance_feature_mtx, X_useridx


def create_dataset(dict_, eid, stance_vt_dict, threshold=90, resolution='day',
                   doc2vec_model=None, user_feature=None, user2ind=None,
                   cutoff=50):
    messages = dict_[eid]
    ts = np.array(messages['timestamps'], dtype=np.int32)
    try:
        user_list = messages['uid'].tolist()
    except:
        user_list = messages['uid']

    x_text, x_stance, XX_uidx = get_features(eid, ts, stance_vt_dict=stance_vt_dict, threshold=threshold, resolution=resolution,
                                             doc2vec_model=doc2vec_model,
                                             user_feature=user_feature, user2ind=user2ind, user_list=user_list,
                                             cutoff=cutoff)

    y = int(messages['label'])
    return x_text, x_stance, XX_uidx, y


if __name__ == "__main__":
    # Setting for CSI
    use_temporal = True
    data_percentage = 90

    dict_ = load_or_create_dataset(CSI_ROOT, use_temporal)

    nb_messages, eid_list, user_set, list_lengths = get_stats(dict_)

    print("# events : {}".format(len(eid_list)))
    print("# users : {}".format(len(user_set)))
    print("# messages : {}".format(np.sum(nb_messages)))
    print("Avg. time length : {} sec\t{} hours".format(np.mean(list_lengths), np.mean(list_lengths) / 3600))
    print("Avg. # messages : {}".format(np.mean(nb_messages)))
    print("Max # messages : {}".format(np.max(nb_messages)))
    print("Min # messages : {}".format(np.min(nb_messages)))
    print("Avg. messages / each user : {}".format(np.sum(nb_messages) / len(user_set)))

    threshold = 50000
    u_sample = get_Usample(dict_, most_common=threshold)
    u_pop = get_Usample(dict_, most_common=len(user_set))
    print("u_sample for most common {} users is obtained.".format(threshold))
    print("u_pop for all {} users is obtained.".format(len(user_set)))

    user_sample2ind = {}
    for ii, (uid, nb_occur) in enumerate(u_sample):
        user_sample2ind[uid] = ii
    print("# users in u_sample : {}".format(len(user_sample2ind)))
    user2ind = {}
    for ii, uid in enumerate(user_set):
        user2ind[uid] = ii
    print("# users : {}".format(len(user2ind)))
    eid2ind = {}
    for ii, eid in enumerate(eid_list):
        eid2ind[eid] = ii
    print("# events : {}".format(len(eid2ind)))

    matrix_sub, eid_sample2ind = get_user_event_matrix(dict_, u_sample, user_sample2ind, binary=True)
    matrix_main, eid_main2ind = get_user_event_matrix(dict_, u_pop, user2ind, binary=True)
    matrix_main_cnt, eid_main_cnt2ind = get_user_event_matrix(dict_, u_pop, user2ind, binary=False)
    print("matrix_sub shape : {}".format(matrix_sub.shape))
    print("Sparsity : {}".format(matrix_sub.count_nonzero() / (matrix_sub.shape[0] * matrix_sub.shape[1])))
    print("matrix_main shape : {}".format(matrix_main.shape))
    print("Sparsity : {}".format(matrix_main.count_nonzero() / (matrix_main.shape[0] * matrix_main.shape[1])))

    u_main_path = os.path.join(CSI_ROOT, "tweet_u_main.npy")
    sigma_main_path = os.path.join(CSI_ROOT, "tweet_sigma_main.npy")
    vt_main_path = os.path.join(CSI_ROOT, "tweet_vt_main.npy")

    u_sub_path = os.path.join(CSI_ROOT, "tweet_u_sub.npy")
    sigma_sub_path = os.path.join(CSI_ROOT, "tweet_sigma_sub.npy")
    vt_sub_path = os.path.join(CSI_ROOT, "tweet_vt_sub.npy")

    RELOAD = False
    if RELOAD:
        # Load matrix_main
        u_main = np.load(open(u_main_path, 'rb'))
        sigma_main = np.load(open(sigma_main_path, 'rb'))
        vt_main = np.load(open(vt_main_path, 'rb'))

        user_feature = u_main.dot(np.diag(sigma_main))
        nb_feature_main = 20  # 10 for weibo, 20 for tweet
        print("user_feature shape : {}".format(user_feature.shape))

        # Load matrix_sub
        u_sub = np.load(open(u_sub_path, 'rb'))
        sigma_sub = np.load(open(sigma_sub_path, 'rb'))
        vt_sub = np.load(open(vt_sub_path, 'rb'))

        user_feature_sub = u_sub.dot(np.diag(sigma_sub))
        nb_feature_sub = 50
        print("user_feature_sub shape : {}".format(user_feature_sub.shape))
        print("Loading is Done.")
    else:
        nb_feature_main = 20  # 10 for weibo, 20 for tweet
        n_iter = 7  # 15 for weibo, 7 for tweet
        u_main, sigma_main, vt_main = randomized_svd(matrix_main, n_components=100,
                                                     n_iter=n_iter, random_state=42)  # random_state=42
        user_feature = u_main.dot(np.diag(sigma_main))
        print("user_feature shape : {}".format(user_feature.shape))

        nb_feature_sub = 50
        matrix_sub = matrix_sub.dot(matrix_sub.transpose())
        matrix_sub_array = matrix_sub.toarray()
        u_sub, sigma_sub, vt_sub = randomized_svd(matrix_sub, n_components=100,
                                                  n_iter=n_iter, random_state=42)  # random_state=42
        user_feature_sub = u_sub.dot(np.diag(sigma_sub))
        print("user_feature_sub shape : {}".format(user_feature_sub.shape))

        np.save(u_main_path, u_main)
        np.save(sigma_main_path, sigma_main)
        np.save(vt_main_path, vt_main)
        np.save(u_sub_path, u_sub)
        np.save(sigma_sub_path, sigma_sub)
        np.save(vt_sub_path, vt_sub)
        print("SVD is done")

    RELOAD_STANCE_VECTOR = False
    stance_vector_dict_path = os.path.join(CSI_ROOT, "stance_vector_dict.pickle")
    if RELOAD_STANCE_VECTOR:
        with open(stance_vector_dict_path, "rb") as f:
            stance_vector_dict = pickle.load(f)
    else:
        stance_vector_dict = build_stance_vector_dict(eid_list, dict_)
        save_to_pickle(stance_vector_dict, stance_vector_dict_path)

    doc2vec_sentences_path = os.path.join(CSI_ROOT, "doc2vec_sentences.pickle")
    model_path = './doc2vec_model/tweet_doc2vec.model'
    sentences = build_doc2vec_dataset(eid_list, dict_)
    print("Num sentences {}".format(len(sentences)))
    doc_vectorizer = train_doc2vec(model_path, sentences)

    LOAD_MODEL = False
    TASK = "classification"  # "classification"

    scaler_dict = {}
    nb_rumor = 0
    burnin = 0

    ### Create dataset ###
    X_text_dict, X_stance_dict = {}, {}
    X_uidx_dict = {}
    subX_dict = {}
    y_dict = {}

    rumor_user = []
    nonrumor_user = []

    with open(os.path.join(CSI_ROOT, "train_test_{}.json".format(data_percentage)), "rb") as f:
        train_test_val = json.load(f)

    eid_train, eid_val, eid_test = train_test_val["train"], train_test_val["val"], train_test_val["test"]

    # Load news graph features
    user_text_dict = load_from_pickle(os.path.join(NEWS_GRAPH_ROOT, "user_features.pickle"))

    for ii, eid in enumerate(eid_list):
        X_text, X_stance, X_uidx, y = create_dataset(dict_, eid, stance_vt_dict=stance_vector_dict, threshold=90 * 24, resolution='hour',
                                                     doc2vec_model=doc_vectorizer, user_feature=user_feature[:, :nb_feature_main],
                                                     user2ind=user2ind, cutoff=50)

        label = int(dict_[eid]['label'])
        if label == 0:
            nonrumor_user.extend(dict_[eid]['uid'])
        elif label == 1:
            rumor_user.extend(dict_[eid]['uid'])
        #     user_ids.update(dict_[eid]['to_user_id'])
        X_text = X_text.astype(np.float32)
        X_stance = X_stance.astype(np.float32)
        if X_text.shape[0] <= 2 * burnin:  # ignore length<=1 sequence
            continue

        X_text_dict[eid], X_stance_dict[eid] = X_text, X_stance
        X_uidx_dict[eid] = X_uidx
        subX_dict[eid] = get_user_feature_in_event(dict_, eid, u_sample,
                                                   user_feature_sub[:, :nb_feature_sub],
                                                   user_sample2ind, user_text_feature=user_text_dict)
        y_dict[eid] = y

        try:
            scaler_dict[eid]
        except:
            # use either X_text or X_stance should be fine
            scaler_hist = MinMaxScaler(feature_range=(0, 1))
            scaler_hist.fit(X_text[:, 0].reshape(-1, 1))
            scaler_interval = MinMaxScaler(feature_range=(0, 1))
            scaler_interval.fit(X_text[:, 1].reshape(-1, 1))
            scaler_dict[eid] = (scaler_hist, scaler_interval)
    print("Dataset are created.")

    params = {
        "task": TASK,
        "nb_feature_sub": nb_feature_sub,
        "burnin": burnin
    }

    save_to_pickle(eid_list, EID_LIST_PATH)
    save_to_pickle(X_text_dict, X_TEXT_DICT_PATH)
    save_to_pickle(X_stance_dict, X_STANCE_DICT_PATH)
    save_to_pickle(y_dict, Y_DICT_PATH)
    save_to_pickle(dict_, DICT_PATH)
    save_to_pickle(subX_dict, SUBX_DICT_PATH)
    save_to_pickle(scaler_dict, SCALER_DICT_PATH)
    save_to_pickle(params, PARAM_PATH)
    save_to_pickle(user2ind, USER2IND_PATH)
    save_to_pickle(eid2ind, EID2IND_PATH)
    save_to_pickle(eid_train, EID_TRAIN_PATH)
    save_to_pickle(eid_val, EID_VAL_PATH)
    save_to_pickle(eid_test, EID_TEST_PATH)
