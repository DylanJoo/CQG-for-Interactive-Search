import re
import json
import numpy as np

f = open('data/canard_provenances_tc.jsonl', 'r')
# "question": clariq_dict['initial_request'],
# "facet": clariq_dict['facet_desc'],
# "c_need": clariq_dict['clarification_need'],
# "c_question": clariq_dict['question'],
# "c_answer": clariq_dict['answer'],
# "q_serp": clariq_serp[clariq_dict['initial_request']],
# "f_serp": clariq_serp[clariq_dict['facet_desc']],

oc_counts = []
ot_counts = []
for i, line in enumerate(f):
    data = json.loads(line.strip())
    # qa
    q = data['question']
    cq = data['c_question']
    ca = data['c_answer']

    # serp
    q_serp, q_serp_scores = data['q_serp']
    qt = set([re.sub(r"\d+", "", t) for t in q_serp])

    try:
        cqt = set([re.sub(r"\d+", "", t) for t in cq_serp])
        cq_serp, cq_serp_scores = data['cq_serp']
    except:
        cqt = qt
        cq_serp = q_serp

    ## overlapped
    oc = [c for c in q_serp if c in cq_serp]
    oc_counts.append(len(oc))
    ot = [t for t in qt if t in cqt]
    ot_counts.append(len(ot))

print(np.mean(oc_counts))
print(">0\t", len([i for i in oc_counts if i > 0]))
print(">=5\t", len([i for i in oc_counts if i >= 5]))
print(">=10\t", len([i for i in oc_counts if i >= 10]))


print(np.mean(ot_counts))
print(">0\t", len([i for i in ot_counts if i > 0]))
print(">=5\t", len([i for i in ot_counts if i >= 5]))
print(">=10\t", len([i for i in ot_counts if i >= 10]))


