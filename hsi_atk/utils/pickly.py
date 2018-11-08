import pickle


def pickle_obj(obj, file):
    f = open(file, "wb")
    pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    f.close()


def load_pickle_obj(file):
    f = open(file, "rb")
    obj = pickle.load(file)
    f.close()
    return obj
