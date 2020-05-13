from constants import *
from tqdm import tqdm
import numpy as np
from doc2vec import get_bin_size


def build_stance_vector_dict(eid_list, dict_):
    threshold = THRESHOLD
    resolution = RESOLUTION
    stance_vector_dict = {}

    for ii, eid in tqdm(enumerate(eid_list), desc="Building stance vector dict"):
        messages = dict_[eid]
        ts = np.array(messages['timestamps'], dtype=np.int32)
        stance_seq = np.array(messages['stances'])

        binsize = get_bin_size(resolution)
        cnt, bins = np.histogram(ts, bins=range(0, threshold * binsize, binsize))

        nonzero_bins_ind = np.nonzero(cnt)[0]
        nonzero_bins = bins[nonzero_bins_ind]

        for bid, bin_left in enumerate(nonzero_bins):
            bin_right = bin_left + binsize
            bin_stance_vectors = []
            # Collecting text to make doc
            for tid, t in enumerate(ts):
                if t < bin_left:
                    continue
                elif t >= bin_right:
                    break
                else:
                    pass
                stance = stance_seq[tid]
                if stance in CHOSEN_STANCES:
                    stance_idx = CHOSEN_STANCES.index(stance)
                    stance_vector = np.zeros(shape=(len(CHOSEN_STANCES)))
                    stance_vector[stance_idx] = 1
                    bin_stance_vectors.append(stance_vector)
            doc_id = str(eid) + '_%s' % bid
            stance_vector_dict[doc_id] = np.mean(bin_stance_vectors, axis=0) if len(bin_stance_vectors) > 0 \
                else np.zeros(shape=(len(CHOSEN_STANCES)))

    return stance_vector_dict