import pandas as pd
df=pd.read_csv('spam.csv',encoding='latin-1')
#print(df.head())

df=df[['v1','v2']]
#print(df.head())
df.columns=['label','text']
#print(df.head())
#print(df['label'].value_counts())
import string
def clean(text):
    text=text.lower()
    text=''.join([char for char in text if char not in string.punctuation])
    text=''.join([char for char in text if not char.isdigit()])
    return text
df['clean_text']=df['text'].apply(clean)
#print(df.head())

from sklearn.feature_extraction.text import TfidfVectorizer
vector=TfidfVectorizer()
x=vector.fit_transform(df['clean_text'])
y=df['label']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,stratify=y,random_state=42)

from sklearn.naive_bayes import MultinomialNB
model=MultinomialNB(alpha=0.5)
model.fit(x_train,y_train)

from sklearn.metrics import classification_report
pred=model.predict(x_test)
print(classification_report(y_test,pred))

new=['chance to win']
cleaned=[clean(text) for text in new]
x_new=vector.transform(cleaned)
n_pred=model.predict(x_new)
print(n_pred)

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,pred))


'''
#pipeline
import pandas as pd
df=pd.read_csv('spam.csv',encoding='latin-1')
df=df[['v1','v2']]
df.columns=['label','text']
from sklearn.model_selection import train_test_split
x=df['text']
y=df['label']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,stratify=y,random_state=42)

from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import string

def clean(text):
    text=text.lower()
    text="".join([char for char in text if char not in string.punctuation])
    text="".join([char for char in text if not char.isdigit()])
    return text

pipeline=Pipeline([
    ('tfidf',TfidfVectorizer(preprocessor=clean)),
    ('model',MultinomialNB(alpha=0.5))
])
pipeline.fit(x_train,y_train)
pred=pipeline.predict(x_test)

from sklearn.metrics import classification_report,confusion_matrix
print("confusion matrix",confusion_matrix(y_test,pred))
print("classification report:",classification_report(y_test,pred))
new=['this is my name','lottery won']
pred_new=pipeline.predict(new)
print(pred_new)'''