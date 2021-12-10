from collections import OrderedDict
import pickle
import os
from tqdm import tqdm
from PyMultiDictionary import MultiDictionary, DICT_THESAURUS, DICT_SYNONYMCOM, DICT_EDUCALINGO, DICT_WORDNET

def synonym_list(word, dictionary):
    while True:
        try:
            synonyms = dictionary.synonym('en', word, dictionary=DICT_THESAURUS)
            #print(word, synonyms[0:5])
            if len(synonyms) > 0:
                return synonyms[0]
            else:
                return None
        except:
            #print('retrying')
            continue

DATASET_ROOT = './datasets/MSCOCO'
token_path = os.path.join(DATASET_ROOT, 'annotations', 'captions_tokens_count.pkl')
with open(token_path, 'rb') as fp:
    token_count = OrderedDict(pickle.load(fp))

token_count_thresh = 10

unknown_token = '</UNK>'
start_token = '</START>'
end_token = '</END>'

token_dict = OrderedDict()

token_dict.update(
    OrderedDict([(k, token_count[k]) for k in sorted(token_count.keys())
                    if token_count[k] > token_count_thresh])
)

token_count_dict = token_dict

dictionary = MultiDictionary()

# synonym_dict = OrderedDict([(word, synonym_list(word, dictionary)) for word in tqdm(list(token_count_dict.keys()))])


# with open('synonym_dict.pkl', 'wb') as handle:
#     pickle.dump(synonym_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('synonym_dict.pkl', 'rb') as handle:
    synonyms = pickle.load(handle)

print(synonyms)