from constants import *
from utils import *
import numpy as np


if __name__ == "__main__":
    publication = read_csv(os.path.join(NEWS_GRAPH_ROOT, "source_publication.tsv"), True, "\t")
    entities = load_text_as_list(os.path.join(NEWS_GRAPH_ROOT, "entities.txt"))
    sources = [row[0] for row in publication]
    news = [row[1] for row in publication]
    reversed_publication_dict = {row[1]: row[0] for row in publication}
    users = [e for e in entities if e not in sources and e not in news]
    source_features, news_features, user_features = {}, {}, {}
    entities_feature_raw = read_csv(os.path.join(NEWS_GRAPH_ROOT, "entity_features.tsv"),  True, "\t")
    for row in entities_feature_raw:
        features = np.array(row[1:], dtype=np.float)
        if row[0] in users:
            user_features[row[0]] = features
        elif row[0] in sources:
            source_features[row[0]] = features
        elif row[0] in news:
            news_features[row[0]] = features
        else:
            raise ValueError("{} is not a valid entities".format(row[0]))
    news_source_features = {news: np.concatenate([source_features[source], news_features[news]])
                            for news, source in reversed_publication_dict.items()}
    save_to_pickle(user_features, os.path.join(NEWS_GRAPH_ROOT, "user_features.pickle"))
    save_to_pickle(news_source_features, os.path.join(NEWS_GRAPH_ROOT, "news_source_features.pickle"))
