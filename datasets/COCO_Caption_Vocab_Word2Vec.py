import nltk
from nltk.tokenize import NLTKWordTokenizer
from pycocotools.coco import COCO
import os
import pickle
import gensim.downloader
import numpy as np
from collections import OrderedDict, Counter


from COCO_Caption import DATASET_ROOT

from Levenshtein import distance as lev_dist

from tqdm import tqdm


TOP_K = 3


if __name__ == '__main__':
    token_path = os.path.join(DATASET_ROOT, 'annotations', 'captions_tokens_count.pkl')

    if not os.path.isfile(token_path):
        captions_ann_paths = [
            os.path.join(DATASET_ROOT, 'annotations', 'captions_train2014.json'),
            os.path.join(DATASET_ROOT, 'annotations', 'captions_val2014.json'),
        ]

        caption_anns = [COCO(x) for x in captions_ann_paths]

        captions = list()

        for caption_ann in caption_anns:
            anns = [x['caption'].strip().lower() for x in caption_ann.anns.values()]
            captions.extend(anns)

        tokenizer = NLTKWordTokenizer()

        tokenized = [tokenizer.tokenize(x) for x in captions]

        token_set = Counter()

        for tkns in tokenized:
            token_set.update(tkns)

        with open(token_path, 'wb') as fp:
            pickle.dump(token_set, fp)
    else:
        with open(token_path, 'rb') as fp:
            token_set = pickle.load(fp)

    token_set = set(token_set.keys())

    wv = gensim.downloader.load("word2vec-google-news-300")

    vocab_set = set(wv.key_to_index.keys())

    vmax = wv.vectors.max()
    vmin = wv.vectors.min()

    tokens_in_vocab = token_set.intersection(vocab_set)

    tokens_not_in_vocab = token_set.difference(tokens_in_vocab)

    vector_dict = OrderedDict({
        '</START>': -wv.vectors[wv.key_to_index['</s>']],
        '</END>': wv.vectors[wv.key_to_index['</s>']],
        '</UNK>': np.random.uniform(low=vmin/10, high=vmax/10, size=wv.vector_size),
        '': np.random.uniform(low=vmin/10, high=vmax/10, size=wv.vector_size),
    })

    for tkn in tqdm(tokens_in_vocab):
        vector_dict[tkn] = wv.vectors[wv.key_to_index[tkn]]

    lower_case_match_tokens = set([x for x in vocab_set if x.lower() in tokens_not_in_vocab])

    for tkn in tqdm(lower_case_match_tokens):
        vector_dict[tkn.lower()] = wv.vectors[wv.key_to_index[tkn]]

    tokens_not_in_vocab = tokens_not_in_vocab.difference(lower_case_match_tokens)

    hyphen_subset = set([x for x in tokens_not_in_vocab if '-' in x])

    for tkn in tqdm(hyphen_subset):
        word_list = tkn.split('-')
        vecs = list()

        for word in word_list:
            if word in vocab_set:
                vec = wv.vectors[wv.key_to_index[word]]
            else:
                dists = [(lev_dist(word, x), x) for x in vocab_set]
                dists.sort(key=lambda x: x[0])
                x1 = np.array([wv.vectors[wv.key_to_index[x[1]]] for x in dists[:TOP_K]])
                vec = np.mean(x1, axis=0)
            vecs.append(vec)

        x1 = np.array(vecs)
        vec = np.mean(x1, axis=0)
        vector_dict[tkn] = vec

    tokens_not_in_vocab = tokens_not_in_vocab.difference(hyphen_subset)

    # Very very slow: 2 seconds per token, wil take 3+ hours to run
    '''
    for tkn in tqdm(tokens_not_in_vocab):
        dists = [(lev_dist(tkn, x), x) for x in vocab_set]
        dists.sort(key=lambda x: x[0])
        x1 = np.array([wv.vectors[wv.key_to_index[x[1]]] for x in dists[:TOP_K]])
        vec = np.mean(x1, axis=0)
        vector_dict[tkn] = vec
    '''

    for tkn in tqdm(tokens_not_in_vocab):
        vector_dict[tkn] = np.random.uniform(low=vmin, high=vmax, size=wv.vector_size)

    vector_path = os.path.join(DATASET_ROOT, 'annotations', 'captions_tokens_vectors.pkl')

    with open(vector_path, 'wb') as fp:
        pickle.dump(vector_dict, fp)

    print('Done')
