import nltk
import json
import pickle

stopwords = nltk.corpus.stopwords.words('english')

with open('./clariq/train_synthetic.pkl', 'rb') as fin, \
     open('clariq.multiturn.train.synthetic.jsonl', 'w') as fout:

    def keyword_filter(word, pos, excluded=""):
        # filter noun / stopword / exluded (appeared request)
        if (pos in ['NOUN', 'ADJ']) and (word not in stopwords) and (word not in excluded):
            return True
        else:
            return False

    data = pickle.load(fin)

    for i, (record_id, data_dict) in enumerate(data.items()):

        user = []
        system = []
        kw_explicit_need = []
        kw_implicit_need = []

        for turn in data_dict['conversation_context']:
            system.append(turn['question'])
            user.append(turn['answer'])
            if 'no' not in turn['answer']:
                # user's answer
                kw_explicit_need += [w for w in nltk.word_tokenize(turn['answer'])]

        # extract facet keyword candidates (explicit (appeared in init request))
        init_need = data_dict['initial_request']

        # extract facet keyword candidates (explicit (appeared in context))
        pos_tags = nltk.pos_tag(kw_explicit_need, tagset="universal")
        kw_explicit_need = list(set([w for w, p in pos_tags if keyword_filter(w, p, init_need)]))

        # extract facet keyword candidates (implicit (not appeared in context))
        kw_implicit_need = [w for w in nltk.word_tokenize(data_dict['question'])]
        pos_tags = nltk.pos_tag(kw_implicit_need, tagset="universal")
        kw_implicit_need = list(set([w for w, p in pos_tags if keyword_filter(w, p, init_need)]))

        fout.write(json.dumps({
            "record_id": record_id,
            "topic_id": data_dict['topic_id'],
            "facet_id": data_dict['facet_id'],
            "init_request": data_dict['initial_request'],
            "question": data_dict['question'],
            "user_utterances": [data_dict['initial_request']] + user,
            "system_responses": system,
            "explicit_keywords": kw_explicit_need,
            "implicit_keywords": kw_implicit_need
        })+'\n')
        
        if i % 10000 == 0:
            print(f"{i} parsed")
            print(init_need)
            print(kw_explicit_need)
            print(kw_implicit_need)

