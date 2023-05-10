import evaluate

# em = evaluate.load('exact_match')
# em.compute(references=['hello'], predictions=['hello'])
# evaluate.combine(["accuracy", "f1", "precision", "recall"])

bleu = evaluate.load('bleu')
