import pickle


def save_as_pickle(data, path):
    with open(path, "wb") as f:
        pickle.dump(data, f)


def load_from_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)