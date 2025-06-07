from ragas.metrics import answer_relevancy, context_precision, context_recall
from ragas import evaluate
import os

def ragas_evaluate(query, retrieved, llm_json):
    results = evaluate(
        queries=[query],
        answers=[llm_json['summary']],
        contexts=[[q['quote'] for q in retrieved]],
        metrics=[answer_relevancy, context_precision, context_recall]
    )
    return results

# Example usage:
# query = "quotes about hope by oscar wilde"
# retrieved = [{"quote": "We are all in the gutter, but some of us are looking at the stars.", "author": "oscar wilde", "tags": "hope, stars"}]
# llm_json = {"summary": "Oscar Wilde's quotes inspire hope by reminding us to look for the stars even in difficult times."}
# print(ragas_evaluate(query, retrieved, llm_json))
