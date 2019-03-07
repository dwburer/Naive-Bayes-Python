from math import log2

"""
Daniel Burer

Requires accompanying data sets (http://qwone.com/~jason/20Newsgroups/20news-bydate-matlab.tgz) to run.
These should be extracted into a folder "<working directory>/matlab"

"vocabulary.txt" (http://qwone.com/~jason/20Newsgroups/vocabulary.txt)
should be placed in the same directory as this python file.

Usage:

CMD: python burer_naive_bayes.py
"""

# global variables
vocabulary = []

# list of known document classifications, where:
# line number = document
# line value = classification id
train_labels = []
test_labels = []

# list of possible classifications for documents
train_map = []
test_map = []
# list of tuples in the form (docId, wordId, wordCount)
train_data = []
test_data = []

print('Reading data from disk...')

# load vocabulary
with open('vocabulary.txt') as f:
    for line in f:
        vocabulary.append(line.strip())
    vocab_length = len(vocabulary)

# load labels
with open('matlab/train.label') as f:
    for line in f:
        train_labels.append(int(line))

with open('matlab/test.label') as f:
    for line in f:
        test_labels.append(int(line))

# load maps
with open('matlab/train.map') as f:
    for line in f:
        train_map.append(line.split()[0])

with open('matlab/test.map') as f:
    for line in f:
        test_map.append(line.split()[0])

# load data
with open('matlab/train.data') as f:
    for line in f:
        train_data.append(tuple([int(n) for n in line.split()]))

with open('matlab/test.data') as f:
    for line in f:
        test_data.append(tuple([int(n) for n in line.split()]))


def parse_data(labels, category_map, data):
    """
    Take raw document data and categorize in dictionary form, while computing category probabilities.

    :param labels: classification of each document in data
    :param category_map: list of labels by which documents may be classified
    :param data: list of tuples in the format (docId, wordId, wordCount)
    :return: data information
    :type: tuple
    """
    print('Parsing data...')

    num_docs = len(labels)
    documents_by_category = {}
    category_probabilities = {}

    for i, category in enumerate(category_map):
        category_id = i + 1

        # document is a list (bag) of words
        for entry in data:
            document_id = entry[0]
            word_id = entry[1]
            word_count = entry[2]

            # if the entry's associated document is of the current category
            if labels[document_id - 1] == category_id:
                if category not in documents_by_category:
                    documents_by_category[category] = {}

                if document_id in documents_by_category[category]:
                    documents_by_category[category][document_id][word_id] = word_count
                else:
                    documents_by_category[category][document_id] = {word_id: word_count}

        category_probabilities[category] = len(documents_by_category[category]) / num_docs

    # create word bags
    list_of_document_bags, list_of_document_label_ids = collect_documents(data, labels)

    return category_probabilities, documents_by_category, list_of_document_bags, list_of_document_label_ids


def calc_word_probabilities(documents_by_category, category_map, alpha=1/vocab_length):
    """
    Calculate the probabilities of each words, given a certain category.

    :param dict documents_by_category:
    :param list category_map: list of labels by which documents may be classified
    :param float alpha: alpha value for Laplace Smoothing
    :return: dictionary of all words probabilities given a category, for each category
    :rtype: dict
    """
    print('Training classifier...')

    word_probability_given_category = {}

    for category in category_map:

        word_occurrences_in_category = {}
        num_words_in_category = 0

        # count number of all words in a category
        for document_word_list in documents_by_category[category].values():
            num_words_in_category += sum(document_word_list.values())

        # count number of occurrences of any given word in a category
        word_probability_given_category[category] = {}

        for doc_id, document_word_list in documents_by_category[category].items():
            for word_id, word_count in document_word_list.items():
                if word_id in word_occurrences_in_category:
                    word_occurrences_in_category[word_id] += word_count
                else:
                    word_occurrences_in_category[word_id] = word_count

        for i, word in enumerate(vocabulary):
            word_id = i + 1

            if word_id in word_occurrences_in_category:
                word_count_in_cat = word_occurrences_in_category[word_id]
            else:
                word_count_in_cat = 0

            num_words_in_cat = num_words_in_category
            v_length = vocab_length
            word_probability_given_category[category][word_id] = (word_count_in_cat + alpha) / (num_words_in_cat + (alpha * v_length))

    return word_probability_given_category


def collect_documents(data, labels):
    """
    Organizes data and returns a list of documents in 'bag of words' dict form.

    :param list data: list of tuples in the format (docId, wordId, wordCount)
    :param list labels: classification of each document in data
    :return: list of documents in "bag of words" form accompanied by label id list
    :rtype: tuple
    """
    document_bag_list = []
    document_bag_label_id_list = []

    current_doc = 1
    current_doc_dict = {}

    for document_id, word_id, word_count in data:
        document_label_id = labels[document_id - 1]

        if document_id == current_doc:
            current_doc_dict[word_id] = word_count
        else:
            document_bag_list.append(current_doc_dict)
            document_bag_label_id_list.append(document_label_id)

            current_doc += 1
            current_doc_dict = {word_id: word_count}

    return document_bag_list, document_bag_label_id_list


def guess(input_document_bag, category_probabilities, word_probability_given_category, category_map):
    """
    Try and classify a given "bag of words" document.

    :param dict input_document_bag: "bag of words" document
    :param dict category_probabilities: probability of each category
    :param dict word_probability_given_category: probabilities of a word given a certain category
    :param list category_map: list of labels by which documents may be classified
    :return: the best guess as to the input document's classification per naive bayes (category id)
    :rtype: int
    """
    probabilities = {}

    for i, category in enumerate(category_map):
        probability_of_category = log2(category_probabilities[category])

        for word_id, word_count in input_document_bag.items():
            for _ in range(word_count):
                word_probability = log2(word_probability_given_category[category][word_id])
                probability_of_category += word_probability

        probabilities[i + 1] = probability_of_category

    return max(probabilities, key=probabilities.get)


def score(train_data_set, train_label_set, test_data_set, test_label_set, category_map, alpha=1/vocab_length):
    """
    Score a Naive Bayes classifier, trained on training data and tested on testing data.

    :param list train_data_set: data set with which to train the classifier
    :param list train_label_set: train data set accompanying classifications
    :param list test_data_set: data set with which to test the classifier
    :param list test_label_set: test data set accompanying classifications
    :param list category_map: list of labels by which documents may be classified
    :param float alpha: alpha value for Laplace Smoothing
    """
    num_documents = len(test_label_set)
    num_correct = 0

    # train the classifier
    train_category_probabilities, train_docs_by_category, list_of_document_bags_train, list_of_document_label_ids_train = parse_data(train_label_set, category_map, train_data_set)
    train_word_prob_given_cat = calc_word_probabilities(train_docs_by_category, category_map, alpha)

    list_of_document_bags_test, list_of_document_label_ids_test = collect_documents(test_data_set, test_label_set)

    print('Evaluating classifier...')

    # set up confusion matrix
    confusion = []
    for i in range(len(category_map)):
        confusion.append([0 for _ in range(len(category_map))])

    # evaluate the classifier based on test data
    for i, current_test_doc in enumerate(list_of_document_bags_test):
        prediction = guess(current_test_doc, train_category_probabilities, train_word_prob_given_cat, category_map)
        actual_category = list_of_document_label_ids_test[i]
        confusion[prediction - 1][actual_category - 1] += 1

        if prediction == actual_category:
            num_correct += 1

    print('\nConfusion matrix:')
    for c in confusion:
        # rjust() for formatting purposes
        print([str.rjust(str(num), 3) for num in c])

    print('\nTotal documents read: %d' % num_documents)
    print('Number of correct guesses: %d' % num_correct)
    print('Score using alpha %f: %f' % (alpha, num_correct / num_documents))


# perform Naive Bayes multinomial text classification
score(train_data, train_labels, test_data, test_labels, test_map)

# uncomment to test different alpha values
# best observed results  around ~0.06
# score(train_data, train_labels, test_data, test_labels, test_map, alpha=0.06)
