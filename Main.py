import os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

DATA_FOLDER = os.path.join(os.getcwd(), "data")
REF_FOLDER = os.path.join(DATA_FOLDER, "reference texts")
FILE_NAME = "huntrprint.txt"


def create_reference_dict(folder):

    comparison_files = {}
    for comp_file in os.listdir(folder):
        flp = os.path.join(folder, comp_file)
        with open(flp, 'r') as fl:
            comparison_files[comp_file] = fl.read()

    return comparison_files


def check_for_plagiarism(text, ref_texts, ngram_length=4):
    """
    :param text: name of the chosen text file to be checked for plagiarism
    :param ref_texts: dictionary with key-value pairs being name and content of text files which are compared to the chosen file
    :param ngram_length: length of n-grams to be compared
    :return: list of names of text files which were plagiarized
    """
    filepath = os.path.join(DATA_FOLDER, text)
    with open(filepath, 'r') as pl:
        text = [pl.read()]

    n_words = len(text[0].split())

    vectorizer = CountVectorizer(ngram_range=(ngram_length, ngram_length))
    vectorizer.fit(text)
    y = vectorizer.transform(ref_texts.values())

    n_overlaps = np.sum(y.toarray(), axis=1)
    files_to_overlaps = dict(zip(ref_texts.keys(), n_overlaps))

    treshold = int((0.1*n_words)/ngram_length)
    final_list = [name for name in files_to_overlaps.keys() if files_to_overlaps[name] > treshold]

    return final_list

if __name__ == "__main__":

    reference_texts = create_reference_dict(REF_FOLDER)

    names = check_for_plagiarism(FILE_NAME, reference_texts)

    print(names)

