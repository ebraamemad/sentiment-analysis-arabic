from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import pandas as pd
df = pd.read_csv('E:\\ready_data.csv')
# Apply label encoding over the labels
lable_encoder = preprocessing.LabelEncoder()
encoded_labels =lable_encoder.fit_transform(df["class"])
df['class']=encoded_labels

X_train, X_validation, y_train, y_validation=train_test_split(df['tweet'], df['class'], test_size=0.2, random_state=42)
