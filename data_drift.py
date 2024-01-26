import pandas as pd
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metrics import EmbeddingsDriftMetric
from evidently.metrics.data_drift.embedding_drift_methods import model, distance, ratio, mmd
from sentence_transformers import SentenceTransformer


model_miniLM = SentenceTransformer('all-MiniLM-L6-v2')

df = pd.read_json("Annotated_report/admin_100.jsonl",lines = True)
ref_embeddings_1 = model_miniLM.encode(df["text"][:150].tolist())

ref_df_1 = pd.DataFrame(ref_embeddings_1)
ref_df_1.columns = ['col_' + str(x) for x in ref_df_1.columns]

cur_embeddings_1 = model_miniLM.encode(df["text"][150:].tolist())


cur_df_1 = pd.DataFrame(cur_embeddings_1)
cur_df_1.columns = ['col_' + str(x) for x in cur_df_1.columns]

column_mapping_1 = ColumnMapping(
    embeddings={'ner_data': ref_df_1.columns[:100]}
)



ner_summary = Report(metrics = [
    EmbeddingsDriftMetric('ner_data',
                          drift_method = distance(
                              dist = 'euclidean', #"euclidean", "cosine", "cityblock" or "chebyshev"
                              threshold = 0.2,
                              pca_components = None,
                              bootstrap = None,
                              quantile_probability = 0.95
                          )
                         )
])


ner_summary.run(reference_data = ref_df_1[:150], current_data = cur_df_1[150:],
           column_mapping = column_mapping_1)


print(ner_summary.as_dict())


