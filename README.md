<h4># BioData001</h4>
<h2>Biology data set analysis</h2>
<pre>
This is the <u>first</u> experience of data analysis in bioinfomatics.
So, naturally, I need to look up the internet to find out how to handle the dna sequence.

And originally this project was given as an interview assignment, so I had done in a day.
The language used for the analysis is <b><u>Python</u></b>.

<b>Code challenge:</b>
Generate a model in any DL/ML frameworks and use this data to learn to classify
a 60 element DNA sequence into the categories of IE, EI or neither.

<b>Reference:</b>
Data set: Dua, D. and Karra Taniskidou, E. (2017).
UCI Machine Learning Repository [http://archive.ics.uci.edu/ml].
Irvine, CA: University of California, School of Information and Computer Science.

Here is the website that I have gained the information how to analyze DNA data:
Working with DNA sequence data for ML by Thomas Nelson
https://www.kaggle.com/thomasnelson/working-with-dna-sequence-data-for-ml
https://www.kaggle.com/thomasnelson/working-with-dna-sequence-data-for-ml-part-2

[1]Evaluation of Convolutionary Neural Networks Modeling of DNA Sequences
using Ordinal versus one-hot Encoding Method
by Allen Chieng Hoon Choong, Nung Kion Lee
bioRxiv 186965; doi: https://doi.org/10.1101/186965

<a href="#NaiveBayes">1. Naive Bayes Model</a>
2. Decision Tree Model
3. Random Forest Model
4. Multi

<b><u>CODE</u></b>
At first, we need to set up the working directory, then check it is directed the right directory.
<pre><code>
import os
os.chdir("F:/code_challenge")
os.getcwd()
</code></pre>
Then check contents in the directory.
<pre><code>
os.listdir()
</pre></code>
Prerequites
<pre><code>
import numpy as np
import pandas as pd
import re

import matplotlib
import matplotlib.pyplot as plt
</pre></code>
Set the plotting enviroment
<pre><code>
%matplotlib inline
matplotlib.style.use("ggplot")
</pre></code>
Uploda the dataset
<pre><code>
data = pd.read_csv('splice.data', names=['classes', 'donor_numbers', 'DNA_sequences'])
data.head()
</pre></code>
Now we can see sequential data in a column 'DNA_sequences' which is written in uppercase letters.

Convert the DNA sequences to lowercase letters and strip whitespaces, and change any non 'acgt' characters to 'n'
where 'agct' represents four nucloebases like adenine (A), cytosine (C), guanine (G) and thymine (T) and
'n' represents the unknown nucleotides.

The reason to change characters is to reduce the usage of memory consumption and
to encode each nucleotide characters as an ordinal values.
That is, A is represented by 0.25, C by 0.50, G by 0.75, and T by 1.00, respectively.
For the unknown nucleotides, n, its value is 0.00[1].

<pre><code>
my_sequence = (pd.DataFrame(data[['DNA_sequences']]))['DNA_sequences']
data['DNA_sequences'] = my_sequence.str.strip()
data['DNA_sequences'] = my_sequence.str.lower()

data.head()
</pre></code>
Define a function to collect all possible overlapping k-mers of specified length
from any sequence string, default size = 6.
<pre><code>
def getKmers(sequence, size=6):
    return [sequence[x:x+size] for x in range(len(sequence)- size + 1)]
</pre></code>
Convert the training data sequences into short overlapping k-mers of length 4.
<pre><code>
data['words'] = data.apply(lambda x: getKmers(x['DNA_sequences']), axis=1)
data = data.drop('DNA_sequences', axis=1)

data.head()
</pre></code>
Address the input and result data as X and y_data for each.
<pre><code>
data_texts = list(data['words'])

for item in range(len(data_texts)):
    data_texts[item] = ' '. join(data_texts[item])

y_data = data.iloc[:,0].values

data_texts[0]

y_data
</pre></code>
Create the bag of words model using CountVectorizer()
This is equivalent to k-mers counting
The n-gram size of 4 was previously determined by testing
<pre><code>
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(ngram_range=(4, 4))
X = cv.fit_transform(data_texts)

print(X.shape)
</pre></code>
At this point, check we are following the right step.
Draw a bar chart to check the category.
<pre><code>
data['classes'].value_counts().sort_index().plot.bar()
</pre></code>
Check the category of the data.
<pre><code>
data_cat = data['classes'].astype('category')
c = data_cat.values
print(type(c))
print(c.categories)
</pre></code>
Check the category of the data by classes.
<pre><code>
grouped = data.groupby('classes')
print(list(grouped))

# pd.unique(data['classes'])
</pre></code>
Split the data into the training and test set, holding out 20% of the data to test the model.
<pre><code>
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y_data,
                                                   test_size = 0.20,
                                                   random_state=42)
print(X_train.shape)
print(X_test.shape)
</pre></code>
<b><u><h4 id="NaiveBayes">A multinomial naive Bayes classifier<h4></u></b> will be used.
The n-gram size of 4 and a model alpha of 0.1 did the best.
<pre><code>
from sklearn.naive_bayes import MultinomialNB

classifier = MultinomialNB(alpha=0.1)
classifier.fit(X_train, y_train)
</pre></code>
Make a predictions on the test set.
<pre><code>
y_pred = classifier.predict(X_test)
</pre></code>
Look at some model performances
<pre><code>
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))
</pre></code>
<b><u>Decision Tree Model</u></b>
<pre><code>
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn import tree
</pre></code>
Import a decision tree classifier().
<pre><code>
data_clf = DecisionTreeClassifier()
data_clf
</pre></code>
Then fit the data set.
<pre><code>
data_clf = data_clf.fit(X_train, y_train)
data_clf
</pre></code>
Predict the result data
<pre><code>
data_prediction = data_clf.predict(X_test)
data_prediction
</pre></code>
Then look at the model performances.
<pre><code>
print("Confusion matrix\n")
print("EI\tIE\tN")
print(confusion_matrix(y_test, data_prediction))

print(classification_report(y_test, data_prediction))
</pre></code>
<b><u>Random Forest Model</u></b>
Import a random forest classifier().
<pre><code>
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

randfc = RandomForestClassifier(n_estimators=10, max_features=3)
randfc
</pre></code>
Then fit the data set and predict the result data.
<pre><code>
randfc.fit(X_train, y_train)
prediction = randfc.predict(X_test)
print(prediction)
print(prediction == y_test)
</pre></code>
Check the accuracy score.
<pre><code>
randfc.score(X_test, y_test)
</pre></code>
Then look at the model performances.
<pre><code>
print(classification_report(prediction, y_test))
</pre></code>
<b><u>Multiple Neural Networks</b></u>
Import a multiple neural networks classfier().
<pre><code>
from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(hidden_layer_sizes=(10, 10, 10))
</pre></code>
Then fit the data set. 
<pre><code>
mlp.fit(X_train, y_train)
</pre></code>
Predict the result data.
<pre><code>
predictions = mlp.predict(X_test)
</pre></code>
Then look at the model performances.
<pre><code>
print("Confusion matrix\n")
print("EI\tIE\tN")
print(confusion_matrix(y_test, predictions))

print(classification_report(y_test, predictions))
</pre></code>

<b><u>Citation Request</u></b>
If you use anything obtained from this repository, then, in your acknowledgements,
please note the assistance you received by using this repository.
Thank you.
</pre>
