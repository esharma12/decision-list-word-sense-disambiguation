'''
Esha Sharma 03/11/24 CMSC 416 PA3
Decision List Classifier 

My decision list contains unigrams from the training data that are ranked according to their
calculated log-likelihood ratios. The decision is a dictionary where the key is in the format: feature , sense. The value of each key is the log-likelihood ratio
associated with the feature. For example the first two key,value pairs in the decision list dictionary is "call , phone: 7.438383530044307" and "introduced , product: 7.438383530044307".
Baseline accuracy is 42.9%. Overall accuracy of tagging is 81.75%
Confusion Matrix:
               phone    product


phone          64       15
product        8        39
1) The problem to be solved is to train a model that will assign a sense, 'phone' or 'product', to the word 'line' in a sentence. 
Essentially, the goal is to correctly ascertain the sense of a word using contexutal features in the training data itself. First, we have an ambiguous word with context, and after the word sense 
disambiguation algorithm is employed, then we can determine the sense of the ambiguous word. In this program's case, the ambigious is line/lines, the context is the bag of words, or unigrams, surrounding line/lines, 
and the two expected senses are 'product' and 'phone'. The training data consists of sentences/paragraphs of information of a specific context separated by <context> tags, and tagged with one of the two senses.
In order to disambiguate the sense of line/lines, a feature vector is created for each sense which consists of unigrams that occur in the training text and are tagged to each sense. 
The WSD algorithm then calculates the log-likelihood ratio of the unigram feature and sense attached to it, ranks the feature decision list in descending order. Then the ratios are used to 
tag the words line/lines in other pieces of text in the testing data.  
in
2) Usage instructions: python3 wsd.py line-train.txt line-test.txt my-model.txt > my-line-answers.txt
The usage instructions listed above is the example input for wsd.py. The program runs using python 3. line-train.txt line-test.txt are txt files that consist of the training and testing data, respectively. my-model.txt 
is the txt file where the decision list model learned by the program is outputted. my-line-answers.txt is where the output of answer tags consisting of the sense tagged to the pieces of testing text is outputted.
Examples of output when the usage instructions are inputted are the following:
my-model.txt:
call , phone: 7.438383530044307
introduced , product: 7.438383530044307
call , product: 7.438383530044307
car , product: 7.3777589082278725...
my-line-answers.txt:
<answer instance="line-n.w8_059:8174:" senseid="phone"/>
<answer instance="line-n.w7_098:12684:" senseid="phone"/>
<answer instance="line-n.w8_106:13309:" senseid="phone"/>
<answer instance="line-n.w9_40:10187:" senseid="phone"/>...
3) The algorithm I used to perform word sense disambiguation is described by the following. First, in order to generalize the program such that any two senses can be used, I
found all of the senses in the training data using regex, and then used the Counter module to count the number of each of the senses. This dictionary also allowed me to grab the specific 
senses and store them in strings. Then, I calculated the most frequent sense that occured in the training data, which I use for my most frequent sense baseline. Then, I extract all of the features 
in the training data and separated them based on the sense they were associated with into two lists. In order to do this, I looped through the training data, and collected all of the sense1 and sense2 data until I reached the </context> tag.
Once the features were extracted, I cleaned up the list of features in the main method using re modules where I removed all numbers and puncuation from the feature lists before I went on to remove stopwords from the lists.
To remove the stopwords, I used the nltk stopword, and I extended it to include the names of the tags I inadvertenly grabbed, like <s> and <p> and <context>, punctuation, and some more english words. Then, I used the Counter module to
calculate the frequency of all unique features, these frequencies were vital in the log likelihood calculations. I also combined both lists and used the Counter module on that list to create a dictionary comprised of all features and their frequencies. 
Then, I went on the calculated the log-likelihood ratios. To do this, I did it twice, for each sense. I looped through the dictionaries containing the features and their frequencies, I calculated the probability of the sense given its feature, Then I checked if the other sense dictionary 
contained that specfic feature, if it did not, then I set the frequency of that features to 0.01 and then calculated the probability of the other sense given the same feature. If the other sense dictionary did contain the same feature and its frequency, then I calculated the probabiltiy
using that value. Then, I performed the log likelihood calculation using the two probabilities, if the ratio was 0, then I smoothed it to 0.01. I added the feature, sense as a key in a decision list dictionary, and the log likelihood ratio as the value. I sorted the decision list dictionary in descending order.
Then, I wrote my decision list model to my-model.txt. With the decision list dictionary created, I read in the testing data, extracted all of the instance ids into a list, and all of the testing text/data into a list using the context tags.
I cleaned up the testing text by removing all digits, punctuations, all words in the stopword list. Finally, I looped through all of the contextual paragraphs of test data and I inner looped through all of the feature/sense keys in the sorted decision list, 
and assigned senses to the word 'line' in the context. To do this, I searched for the current feature in the sorted decision list dictionary in the current context, once a feature in the sorted list was found in the current context, I assigned the sense associated with the found feature to the variable, current_sense.
If a feature in the sorted list could not be found in the current context, then I set current_sense to the most frequent sense. Finally, within the inner loop, I printed the answer tag out to my-line-answers.txt using the instance ids in test_ids list and the found current sense. 
'''
from pprint import pprint
from collections import Counter
from sys import argv
import re
import math
from nltk.corpus import stopwords 
import operator

def main():
    #save file names from command line
    train_file = str(argv[1])
    test_file = str(argv[2])
    model_file = str(argv[3])

    #open and read test file
    file = open(train_file, 'r')
    train_data = file.read()
    train_text = ''.join(train_data)
    #find all senses and count frequencies
    senses = re.findall(r'senseid="(.*)"/>', train_text)
    counted_senses = dict(Counter(senses))
    #grab generalized senses strings and their frequences
    sense1, sense2 = counted_senses.keys()
    sense1_freq, sense2_freq = counted_senses.values()

    #calculate most frequent sense in training data for baseline accuracy
    most_freq_sense = ""
    if sense1_freq >= sense2_freq:
        most_freq_sense = sense1
    else:
        most_freq_sense = sense2
        
    #returns list of uncleaned features according to sense
    sense1_features = extractFeatures1(train_text, sense1)
    sense2_features = extractFeatures2(train_text, sense2)

    #remove puncuations and digits from feature list for sense1
    features1_string = ' '.join(sense1_features).lower()
    features1_string = re.sub(r'[0-9]', '', features1_string)
    sense1_features = re.findall(r"[\w']+|[.,!?;]", features1_string)
    features1_string = ' '.join(sense1_features)

    #remove puncuations and digits from feature list for sense2
    features2_string = ' '.join(sense2_features).lower()
    features2_string = re.sub(r'[0-9]', '', features2_string)
    sense2_features = re.findall(r"[\w']+|[.,!?;]", features2_string)
    features2_string = ' '.join(sense2_features)
   
    #download nltk stopword list and extend it to include tags in training data and puncuation
    stop_words = stopwords.words('english')
    stop_words.extend([".", ",", "s", "p", "context", "senseid", ";", "-", "--", "!", "?", "'", "'s", "head"])
    stopwords_dict = Counter(stop_words)
    #use list comprehension to remove stopwords from sense1 features
    features1_string = ' '.join([word for word in features1_string.split() if word not in stopwords_dict])
    sense1_features = features1_string.split()
    #use list comprehension to remove stopwords from sense2 features
    features2_string = ' '.join([word for word in features2_string.split() if word not in stopwords_dict])
    sense2_features = features2_string.split()

    #consolidate and count frequencies of features for sense1 and sense2
    counted_sense1_features = Counter(sense1_features)
    counted_sense2_features = Counter(sense2_features)

    #combine features for sense1 and sense2 into one list and count frequencies to be used later in calculations
    total_features = sense1_features + sense2_features
    counted_total_features = Counter(total_features)
    final_feature_list = {}

    #create final feature list for both senses, with feature and sense as keys, and log likelihood ratios as values
    final_feature_list = calculateSense1Ratio(final_feature_list, counted_total_features, counted_sense1_features, counted_sense2_features, sense1)
    final_feature_list = calculateSense2Ratio(final_feature_list, counted_total_features, counted_sense1_features, counted_sense2_features, sense2)

    #sort final decision list in descending order to help when senses needs to be picked for context data in the test file
    sorted_final_feature_list = dict(sorted(final_feature_list.items(), key=operator.itemgetter(1), reverse=True))
    
    #write the decision list model to my-model.txt in the form of "feature, sense: log-likelihood ratio"
    with open(model_file, "w") as file:
        for key in sorted_final_feature_list:
            file.write(str(key) + ": " + str(sorted_final_feature_list[key]) + "\n")

    #open and read in test file into string
    f = open(test_file, 'r')
    test_data = f.read()
    test_text = ''.join(test_data)
    #extract all instance ids and save them into a list
    test_ids = re.findall(r'id="(.*):">', test_text)
    #extract all context test data and save into list, separated based on instance id
    test_contexts = re.findall(r'<context>\n(.*)\n</context>', test_text)

    for item in test_contexts:
        #clean context data, removing digits and puncutation and stopwords, and then reinstantiating the list with cleaned up string
        item_context_string = item.lower()
        item_context_string = re.sub(r'[0-9]', '', item_context_string)
        item_context = re.findall(r"[\w']+|[.,!?;]", item_context_string)
        item_context_string = ' '.join(item_context)
        item_context_string = ' '.join([word for word in item_context_string.split() if word not in stopwords_dict])
        i = test_contexts.index(item)
        test_contexts[i] = item_context_string

    #with cleaned up test data, assign sense based on sorted feature decision list
    for context in test_contexts:
        #inner loop for all features in sorted dictionary
        for key in sorted_final_feature_list:
            #extract feature and sense of current key
            feature_sense = key.split(" , ")
            feature = feature_sense[0]
            sense = feature_sense[1]
            #search for feature in the current context string, if feature is found, assigned sense associated with found feature in dictionary, and stop searching immediately
            if re.search(rf"\b{feature}\b", context):
                current_sense = sense
                break
        #if whole dictionary has been iterated through and no sense has been assigned, then assign most frequent sense from training data
        if current_sense == "":
            current_sense = most_freq_sense
        #find index value to be able to output instance id of the context string currently worked on
        i = test_contexts.index(context)
        #print out answer tag with instance id and current assigned sense, in the format of line-key.txt
        print('<answer instance="' + test_ids[i] + ':" senseid="' + current_sense + '"/>')
  

def extractFeatures1(train_text, sense1):
    sense1_features = [] 
    train_text_split = train_text.split()
    #to help with generalization, using sense1 variable 
    p1 = 'senseid="%s"/>' % (sense1)
    for i in range(0, len(train_text_split)):
        #once we reach the senseid tag, start iterating through the training text again until the </context> tag is found
        if train_text_split[i] == p1:
            for j in range(i, len(train_text_split)):
                if train_text_split[j] != '</context>':
                    #add all features found for sense1 to sense1_features list
                    sense1_features.append(train_text_split[j])
                else:
                    #when senseid tag is not associated to sense1, jump to j and start searching again
                    i = j
                    break
    return sense1_features

def extractFeatures2(train_text, sense2):
    sense2_features = []
    train_text_split = train_text.split()
    #to help with generalization, using sense2 variable
    p2 = 'senseid="%s"/>' % (sense2)
    for i in range(0, len(train_text_split)):
        #once we reach the senseid tag, start iterating through the training text again until the </context> tag is found
        if train_text_split[i] == p2:
            for j in range(i, len(train_text_split)):
                if train_text_split[j] != '</context>':
                    #add all features found for sense1 to sense2_features list
                    sense2_features.append(train_text_split[j])
                else:
                    #when senseid tag is not associated to sense2, jump to j and start searching again
                    i = j
                    break
    return sense2_features

def calculateSense1Ratio(final_feature_list, counted_total_features, counted_sense1_features, counted_sense2_features, sense1):
    prob_key_sense1 = 0
    prob_key_sense2 = 0
    for key in counted_sense1_features:
        #calculate probability of sense1 given feature
        prob_key_sense1 = counted_sense1_features[key] / counted_total_features[key]
        #if same feature is not in the other list of sense2 features then smooth frequency by setting it to 0.01 and then calculate probability 
        if key not in counted_sense2_features:
            counted_sense2_features[key] = 0.01
            prob_key_sense2 = counted_sense2_features[key] / counted_total_features[key]
        else:
            #if feature in list of sense2 features, calculate probability using frequency value
            prob_key_sense2 = counted_sense2_features[key] / counted_total_features[key]
        #calculate log likelihood ratio 
        ratio = abs(math.log(prob_key_sense1/prob_key_sense2))
        #if ratio 0, then smooth it with a 0.01
        if ratio == 0.0:
            ratio = 0.01
        #save calculated ratio in the form key, sense1: ratio
        final_feature_list[key + " , " + sense1] = ratio

    return final_feature_list    

def calculateSense2Ratio(final_feature_list, counted_total_features, counted_sense1_features, counted_sense2_features, sense2):
    prob_key_sense2 = 0
    prob_key_sense1 = 0
    for key in counted_sense2_features:
        #calculate probability of sense2 given feature
        prob_key_sense2 = counted_sense2_features[key] / counted_total_features[key]
        #if same feature is not in the other list of sense1 features then smooth frequency by setting it to 0.01 and then calculate probability 
        if key not in counted_sense1_features:
            counted_sense1_features[key] = 0.01
            prob_key_sense1 = counted_sense1_features[key] / counted_total_features[key]
        else:
            #if feature in list of sense1 features, calculate probability using frequency value
            prob_key_sense1 = counted_sense1_features[key] / counted_total_features[key]
        #calculate log likelihood ratio 
        ratio = abs(math.log(prob_key_sense2/prob_key_sense1))
        #if ratio 0, then smooth it with a 0.01
        if ratio == 0.0:
            ratio = 0.01
        #save calculated ratio in the form key, sense2: ratio
        final_feature_list[key + " , " + sense2] = ratio

    return final_feature_list

if __name__ == "__main__":
    main()