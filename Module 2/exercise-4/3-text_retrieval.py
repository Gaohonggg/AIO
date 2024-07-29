import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

vi_data_df = pd.read_csv("vi_text_retrieval.csv")
context = vi_data_df["text"]
context = [doc.lower() for doc in context]

tfidf_vectorizer = TfidfVectorizer()
context_embedded = tfidf_vectorizer.fit_transform(context).toarray()

def tfidf_search(question, tfidf_vectorizer, top_d=5):
    query_embedded = tfidf_vectorizer.transform([question.lower()]).toarray()
    cosine_scores = cosine_similarity(query_embedded, context_embedded).flatten()
    results = []
    for i in cosine_scores.argsort()[-top_d:][::-1]:
        doc_score = {
            'id': i,
            'cosine': cosine_scores[i]
        }
        results.append(doc_score)
    return results

question = vi_data_df.iloc[0]["question"]
results = tfidf_search(question, tfidf_vectorizer, top_d=5)
print(results[0]['cosine'])

def corr_search(question, tfidf_vectorizer, top_d=5):
    query_embedded = tfidf_vectorizer.transform([question.lower()]).toarray().flatten()
    context_embedded_dense = context_embedded
    corr_scores = np.array([np.corrcoef(query_embedded, doc_embedded)[0, 1] for doc_embedded in context_embedded_dense])
    results = []
    for idx in corr_scores.argsort()[-top_d:][::-1]:
        doc = {
            'id': idx,
            'corr_score': corr_scores[idx]
        }
        results.append(doc)
    return results

question = vi_data_df.iloc[0]["question"]
results = corr_search(question, tfidf_vectorizer, top_d=5)
print(results[1]['corr_score'])
