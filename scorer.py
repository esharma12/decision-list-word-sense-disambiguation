'''
Esha Sharma 03/24/24 CMSC 416 PA4
Naive Bayes Model Accuracy and Confusion Matrix:            Logistic Regression Model Accuracy and Confusion Matrix:            Stochastic Gradient Descent Model Accuracy and Confusion Matrix:
Accuracy is: 93.65079365079364%                             Accuracy is: 81.74603174603175%                                     Accuracy is: 79.36507936507937%

         phone  product                                                phone  product                                                    phone  product
                       
phone       69        5                                    phone       55        6                                              phone       48        2
product      3       49                                    product     17       48                                              product     24       52

In my PA3 program, my baseline accuracy was 42.9% and my overall accuracy of tagging was 81.75% In PA3, I used a ranked decision list with log-likelihood ratios. My confusion matrix from PA3 is below:
Confusion Matrix:
               phone    product


phone          64       15
product        8        39
1) The problem to be solved is to write a utility program that will compare the outputted answer tags in my-line-answers.txt to the answer tags in the provided file, line-key.txt.
For generalization purposes, the senses used to perform word sense disambiguation in wsd-ml.py will be extracted from line-key.txt, as well as counted to provide total number answer tags. 
Additionally, after comparing and calculating frequency of every possible result, where sense1 was the expected answer and sense1 is the real answer, where sense1 was the expected answer but sense2
is the real answer, where sense2 was the expected answer and sense2 is the real answer, and finally where sense2 was the expected answer and sense1 is the real answer. The final accuracy is outputted by
this program, as well as the confusion matrix containing all of the values of the four possible result listed above. To create a confusion matrix, the pandas module was used.  
2) Usage instructions: python3 scorer.py my-line-answers.txt line-key.txt
For input of this program, wsd-ml.py will have to be run first so the most current version of my-line-answers.txt, associated with the ml-model implemented in wsd-ml.py, is available for accuracy calculations. The example input is the same
as the usage instructions listed above. The program is run using python3. scorer.py is the name of this program, my-line-answers.txt is the file that wsd-ml.py outputs the predicted answer tags to, and 
line-key.txt is the provided file that contains the standard "key" data. An example output is the following:

Accuracy is: 93.65079365079364%

         phone  product
                       
phone       69        5
product      3       49

3) The algorithm I used to calculate my total accuracy and confusion matrix for the output of wsd-ml.py is described by the following. First, I read in the file names from the command line.
The answer tags from my-line-answers.txt are read in to a list, and the answer tags from line-key.txt are also read into a list. To generalize the program, 
I use regex to find the two senses in the key file, count them to calculate the total. Then, I looped through the key answer tags. I checked if answer tags were equal, if so I incremented a counter variable, 
and then checked if the senseid in the answer tag was equal to the sense1 variable, if so, I appended 'sense1' to both the real_answers and expected_answers lists. If the senseid was equal to the sense2 variable, then 
I appended 'sense2' to both the real_answers and expected_answers lists. If the answers tag are not equal, then I enter an else statement. If the senseid in the key answers tag was equal to sense1, then I append sense1 to 
the real_answers list and sense2 to the expected_answers list. If the senseid in the key answer tag was equal to sense2, then I append sense2 to the real_answers list and sense1 to the expected answers list
Then, I calculate and print out the accuracy using the counter variable containing number of correct answers and the total number of answer tags in the key file. Finally, I created two pandas module series arrays
containing the senses in the real_answers and expected_answers lists. Then, I created a confusion matrix with the expected and real counts and printed the matrix out. 
'''

import re
import pandas as pd
from sys import argv
from collections import Counter

#read in the file names from the command lines
model_answers_file = str(argv[1])
key_answers_file = str(argv[2])

#declare lists and counter variable
model_answers = []
key_answers = []
real_answers = []
expected_answers = []
num_correct_answers = 0

#open and read in the answers tags from the my-line-answers.txt file to a list
with open(model_answers_file, "r") as file:
    for answer in file:
        model_answers.append(answer)

#open and read in the answers tags from the line-key file to a list
with open(key_answers_file, "r") as file:
    for answer in file:
        key_answers.append(answer)
#convert key list to string
key_string = ''.join(key_answers)

#find all the senses in the key string, count them using Counter, and extract sense1 and sense2 
senses = re.findall(r'senseid="(.*)"/>', key_string)
counted_senses = dict(Counter(senses))
sense1, sense2 = counted_senses.keys()


for i in range(0, len(key_answers)):
        #check if answers tags are equal
        if key_answers[i] == model_answers[i]:
            #if equal, increment counter
            num_correct_answers += 1
            #check if sense1 is in key answer tag
            if re.search(rf'{sense1}', key_answers[i]):
                #if sense1 exists, append sense1 to both real_answers and expected_answers lists
                real_answers.append(sense1)
                expected_answers.append(sense1)
            else:
                #if sense1 is not in key answer tag, append sense2 to both real_answers and expected_answers lists
                real_answers.append(sense2)
                expected_answers.append(sense2)
        #if answers tags are not equal
        else:
            #check if sense1 is in key answer tag
            if re.search(rf'{sense1}', key_answers[i]):
                #if sense1 exists, append sense1 to real_answers list but sense2 to expected_answers list
                real_answers.append(sense1)
                expected_answers.append(sense2)
            else:
                #if sense1 does not exist, append sense2 to real_answers list but sense1 to expected_answers list 
                real_answers.append(sense2)
                expected_answers.append(sense1)

#total is number of key answer tags in the line-key file
total = len(key_answers)
#calculate accuracy using counter for correct answers and total
accuracy = (num_correct_answers / total) * 100
#print accuracy 
print("Accuracy is: " + str(accuracy) + "%")

#create Series arrays with the real and expected senses with the pandas module
real = pd.Series(real_answers, name = '')
expected = pd.Series(expected_answers, name = '')

#create confusion matrix using expected senses and real senses
confusion_matrix = pd.crosstab(expected, real)

#print confusion matrix
print("\n%s" % confusion_matrix)