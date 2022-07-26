


from sklearn import datasets
from utils import Filter
from pandas import pd

iris = datasets.load_iris()
df = pd.DataFrame(data=iris['data'], columns=iris['feature_names'])
df['class_label'] = iris['target']

df_filtered = Filter.filter_by_minmax(df, field='sepal length (cm)', minmax_values=(5., 6.), verbose=True)


