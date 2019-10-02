import numpy as np
import re

from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec


def build_doc2vec_dataset(eid_list, dict_):
    threshold = 90 * 24
    resolution = 'hour'
    sentences = []

    for ii, eid in enumerate(eid_list):
        if ii % 100 == 0:
            print("{}th event {} is processing...".format(ii + 1, eid))
        messages = dict_[eid]
        ts = np.array(messages['timestamps'], dtype=np.int32)
        text_seq = np.array(messages['text'])

        if resolution == 'day':
            binsize = 3600 * 24
        elif resolution == 'hour':
            binsize = 3600
        elif resolution == 'minute':
            binsize = 60
        else:
            raise ValueError("Unrecognized resolution {}".format(resolution))
        cnt, bins = np.histogram(ts, bins=range(0, threshold * binsize, binsize))

        nonzero_bins_ind = np.nonzero(cnt)[0]
        nonzero_bins = bins[nonzero_bins_ind]
        print(ii, eid, len(nonzero_bins))

        for bid, bin_left in enumerate(nonzero_bins):
            bin_right = bin_left + binsize
            try:
                del doc
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
                string = text_seq[tid]
                string = re.sub(r"http\S+", "", string)
                string = re.sub("[?!.,:;()'@#$%^&*-=+/\[\[\]\]]", ' ', string)  # !.,:;()'@#$%^&*-_{}=+/\"
                try:
                    doc += string
                except:
                    doc = string
            if isinstance(eid, int):
                eid_str = str(eid)
            else:
                eid_str = eid
            doc_id = eid_str + '_%s' % bid
            print(doc_id)
            sentences.append(LabeledSentence(utils.to_unicode(doc).split(), [doc_id]))

    print("length of sentences : {}".format(len(sentences)))
    return sentences


def train_doc2vec(model_path, sentences):
    from gensim.models import Doc2Vec

    try:
        doc_vectorizer = Doc2Vec.load(model_path)
        print("doc_vectorizer is loaded.")
    except:
        doc_vectorizer = Doc2Vec(min_count=1, window=10, size=100, sample=1e-4, negative=5, workers=8)
        doc_vectorizer.build_vocab(sentences)
        print("build_vocab is done.")

        for epoch in range(10):
            print(epoch)
            doc_vectorizer.train(sentences,
                                 total_examples=doc_vectorizer.corpus_count,
                                 epochs=doc_vectorizer.epochs)
        print("doc2vec training is done.")
        doc_vectorizer.save(model_path)

    print(doc_vectorizer.docvecs['E356_10'].shape)
    return doc_vectorizer
