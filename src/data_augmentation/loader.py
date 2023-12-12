import json
import unicodedata
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from datasets import load_dataset, Dataset

def clariq(path):
    # Add column (q and cq) 
    df = pd.read_csv(path, delimiter='\t').dropna()
    clariq = Dataset.from_pandas(df)
    clariq = clariq.rename_column('question', 'c_question')
    clariq = clariq.rename_column('initial_request', 'question')
    clariq = clariq.map(lambda ex: 
            {"q_and_cq": f"{ex['question']} {ex['c_question']}"}
    )
    return clariq

def qrecc(path):

    qrecc = load_dataset('json', data_files=path)['train']
    qrecc = qrecc.rename_column('Question', 'utterance')
    qrecc = qrecc.rename_column('Rewrite', 'question')
    qrecc = qrecc.rename_column('Answer', 'answer')
    qrecc = qrecc.map(lambda ex: 
            {"id": f"{ex['Conversation_no']}_{ex['Turn_no']}"}
    )
    ### Here, use rewritten question as query
    ### [NOTE] For dense retrieval, it's possible to use Conv query
    qrecc = qrecc.map(lambda ex: 
            {"q_and_a": f"{ex['question']} {ex['answer']}"}
    )
    return qrecc

def topiocqa(path, reference_key='answer'):
    """ Here we use the utterance as question.  
    the key should be something related to gold passages (Rationale)
    """
    topiocqa = load_dataset('json', data_files=path)['train']
    topiocqa = topiocqa.rename_column('Question', 'question')
    # topiocqa = topiocqa.rename_column('Rewrite', 'question')
    topiocqa = topiocqa.rename_column('Answer', 'answer')
    topiocqa = topiocqa.rename_column('Rationale', 'rationale')
    topiocqa = topiocqa.map(lambda ex: 
            {"id": f"{ex['Conversation_no']}_{ex['Turn_no']}"}
    )
    ### Here, use rewritten question as query
    ### [NOTE] For dense retrieval, it's possible to use Conv query
    topiocqa = topiocqa.map(lambda ex: 
            {"q_and_a": f"{ex['question']} {ex[reference_key]}"}
    )
    return topiocqa

def load_collections(path, title=False, full=True):
    def normalize(text):
        return unicodedata.normalize("NFD", text)

    if full:
        collections = {}
        titles = {}
    else:
        collections = defaultdict(str)
        titles = defaultdict(str)

    print("Load collections...")
    with open(path, 'r') as f:
        for i, line in enumerate(tqdm(f)):
            item = json.loads(line.strip())
            collections[str(item['id'])] = normalize(item['contents'])
            titles[str(item['id'])] = normalize(item['title'])
            if (i == 10000) and (full is False):
                break
    if title:
        return collections, titles
    else:
        return collections

def load_cqg_predictions(path):
    data = {}
    with open(path, 'r') as f:
        for line in tqdm(f):
            item = json.loads(line.strip())
            data[str(item['qid'])] = item['c_question']
    return data
