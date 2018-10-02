<h4># BioData001</h4>
<h2>Biology data set analysis</h2>

This is the <u>first</u> experience of data analysis in bioinfomatics.
So, naturally, I need to look up the internet to find out how to handle the dna sequence.

And originally this project was given as an interview assignment, so I had done in a day.
The language used for the analysis is <b><u>Python</u></b>.

<b>Code challenge:</b>
Generate a model in any DL/ML frameworks and use this data to learn to classify
a 60 element DNA sequence into the categories of IE, EI or neither.

<b>Reference:</b>
Data set: Dua, D. and Karra Taniskidou, E. (2017). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml].
Irvine, CA: University of California, School of Information and Computer Science.
Here is the website that I have gained the information how to analyze DNA data:
https://www.kaggle.com/thomasnelson/working-with-dna-sequence-data-for-ml
https://www.kaggle.com/thomasnelson/working-with-dna-sequence-data-for-ml-part-2

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
Set the plotting enviroment ment
<pre><code>
%matplotlib inline
matplotlib.style.use("ggplot")
</pre></code>
Uploda the dataset
<pre><code>
data = pd.read_csv('splice.data', names=['classes', 'donor_numbers', 'DNA_sequences'])
data.head()
</pre></code>
Convert the DNA sequences to lowercase letters and strip whitespaces, and change any non 'acgt' characters to 'n'
to encode each nucleotide characters as an ordinal values.
<pre><code>
my_sequence = (pd.DataFrame(data[['DNA_sequences']]))['DNA_sequences']
data['DNA_sequences'] = my_sequence.str.strip()
data['DNA_sequences'] = my_sequence.str.lower()

data.head()
</pre></code>
Define a function to collect all possible overlapping k-mers of specified length from any sequence string, default size = 6.
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

<pre><code>

</pre></code>

<pre><code>

</pre></code>

<pre><code>

</pre></code>

<pre><code>

</pre></code>

<pre><code>

</pre></code>

<pre><code>

</pre></code>

<pre><code></pre></code>
