import os

CSI_ROOT = "data/csi"
EID_LIST_PATH = os.path.join(CSI_ROOT, "eid_list.pickle")
X_TEXT_DICT_PATH = os.path.join(CSI_ROOT, "x_text_dict.pickle")
X_STANCE_DICT_PATH = os.path.join(CSI_ROOT, "x_stance_dict.pickle")
X_DICT_STANCE_PATH = os.path.join(CSI_ROOT, "x_dict_stance.pickle")
Y_DICT_PATH = os.path.join(CSI_ROOT, "y_dict.pickle")
DICT_PATH = os.path.join(CSI_ROOT, "dict_.pickle")
SUBX_DICT_PATH = os.path.join(CSI_ROOT, "subX_dict.pickle")
SCALER_DICT_PATH = os.path.join(CSI_ROOT, "scaler_dict.pickle")
PARAM_PATH = os.path.join(CSI_ROOT, "params.pickle")
USER2IND_PATH = os.path.join(CSI_ROOT, "user2ind.pickle")
EID2IND_PATH = os.path.join(CSI_ROOT, "eid2ind.pickle")
EID_TRAIN_PATH = os.path.join(CSI_ROOT, "eid_train.pickle")
EID_VAL_PATH = os.path.join(CSI_ROOT, "eid_val.pickle")
EID_TEST_PATH = os.path.join(CSI_ROOT, "eid_test.pickle")
NEWS_GRAPH_ROOT = "data/news_graph"

THRESHOLD = 90 * 24
RESOLUTION = 'hour'
CHOSEN_STANCES = ["support", "deny"]