# Spam-Classification
# It will classifie the mails as spam and not spam based upon the text which will help us to detect the spam mails and reduce our time to watch unneccessary mails

# code starts for here
# import dataset 
import pandas as pd
df=pd.read_csv("C:\\Users\\RGUKT\\Documents\\RAJKUMAR CODING\\MACHINE LEARNING\\spam.csv")
df
# i got three unnecessary columns if you got then delete it
df.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1,inplace=True)
df
# labeled the names as input and output for clear understanding

df2=pd.DataFrame(df.iloc[1:,:].values,columns=['output','input'])
df2
df=df2.copy()
df
# to know how many spam and ham(not spam) number
df['output'].value_counts()

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
corpus=[]
# TO REMOVE UNNNECCESSARY WORDS AND REPLACE NUMBERS AND SPECIAL CHARACTERS WITH SPACE
import re
for i in range(len(df)):
    try:
        review=re.sub('[^a-zA-Z]',' ',df['input'][i])
        review=review.lower()
        review=review.split()
        stop=set(stopwords.words('english'))
        review=[ps.stem(word) for word in review if  word not in  stop ]
        review=''.join(review)
        corpus.append(review)
    except KeyError as e:
        continue
# to check the length of the corpus array
len(corpus)
# creating BAG of WORDS
from sklearn.feature_extraction.text import CountVectorizer
vector=CountVectorizer()
x=vector.fit_transform(corpus).toarray()

y=pd.get_dummies(df['output'])
y=y.iloc[:,1]
y.replace({True:1,False:0})
# Creating a model and fitting the data
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
mul=MultinomialNB()
mul.fit(x_train,y_train)
y_pred=mul.predict(x_test)
accu=accuracy_score(y_test,y_pred)
accu
# end of the code
