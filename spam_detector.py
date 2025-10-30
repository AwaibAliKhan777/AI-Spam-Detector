import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

#Load the dataset from the internet
data = pd.read_csv("https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv",
                   sep='\t', header=None, names=['label', 'message'])

#convert text labels to numbers (spam = 1 , ham=0)
data['label_num'] = data.label.map({'ham':0,'spam':1})

#Split the data into training and testing parts
X_train,X_test,y_train,y_test = train_test_split(data['message'],data['label_num'],test_size=0.2)


#convert text into numbers (Bag of words)
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

#create and train the model
model = MultinomialNB()
model.fit(X_train_vec,y_train)

#check accuracy
y_pred = model.predict(X_test_vec)
print("Accuracy:" , accuracy_score(y_test,y_pred))

#Test your own message 
msg = input("Enter a message to check:")
msg_vec = vectorizer.transform([msg])
print("Spam" if model.predict(msg_vec)[0] else "Not Spam")