from sklearn.datasets import fetch_openml
data = fetch_openml('adult', version=2, as_frame=True, parser='auto')
print(data.frame.columns.tolist())
