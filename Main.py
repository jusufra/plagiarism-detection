import os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

PL_FILE = os.path.join(os.getcwd(), "bulhuntr.txt")  # this is the chosen file which is to be checked for plagiarism
TXT_FOLDER = os.path.join(os.getcwd(), "texts")  # this folder contains the files which are to be compared with the chosen file

def check_for_plagiarism(text, reference_texts, ngram_length=4):
    """
    :param text: list with a single element, which is the content of the chosen text file to be checked for plagiarism
    :param reference_texts: dictionary with key-value pairs being name and content of text files which are compared to the chosen file
    :return: list of names of text files which were plagiarized
    """
    n_words = len(text[0].split())

    vectorizer = CountVectorizer(ngram_range=(ngram_length, ngram_length))
    vectorizer.fit(text)

    y = vectorizer.transform(reference_texts.values())

    n_overlaps = np.sum(y.toarray(), axis=1)
    files_to_overlaps = dict(zip(reference_texts.keys(), n_overlaps))

    treshold = int((0.1*n_words)/ngram_length)
    final_list = [name for name in files_to_overlaps.keys() if files_to_overlaps[name] > treshold]

    return final_list

if __name__ == "__main__":

    with open(PL_FILE, 'r') as pl:
        file = [pl.read()]

    comparison_files = {}
    for comp_file in os.listdir(TXT_FOLDER):
        fl = os.path.join(TXT_FOLDER, comp_file)
        with open(fl, 'r') as fl:
            comparison_files[comp_file] = fl.read()

    names = check_for_plagiarism(file, comparison_files)

    print(names)

