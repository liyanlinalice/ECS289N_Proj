import word2vec
import xlrd
import nltk
import string
import numpy as np
import pickle

model = word2vec.load('./text8.bin')
dataFrame = open('./reviews_rating300limit_all_balanced.pkl', 'rb')
dataFrame = pickle.load(dataFrame)
reviews = dataFrame['reviews']
stars = dataFrame['stars']
type(stars)
num_reviews = len(stars)
num_reviews_for_a_star = num_reviews / 5

reviews_first_100 = []
REVIEW_NUM = 8000
REVIEW_LEN_LIMIT = 100
review_matrix_list = []
review_scores = []
review_selected_list = []
i_eff_review = 0
vec_len = len(model['love'])
print("Vec_len From Word-2-Vec == ", vec_len)

def get_vecs(i):
    not_found = True
    global i_eff_review

    while True:
        text = reviews[i].lower().translate(str.maketrans("", "", string.punctuation))
        words = nltk.word_tokenize(text)
        words_matrix = []
        num_word_in_dic = 0
        num_word_total = 0
        for word in words:
            num_word_total += 1
            if word in model:
                num_word_in_dic += 1
                words_matrix.append(model[word])
                if num_word_in_dic == REVIEW_LEN_LIMIT:
                    break

        if (num_word_total > 0) and (num_word_in_dic / num_word_total) > 0.8: # a useable review
            i_eff_review += 1
            not_found = False
            review_selected_list.append(i)
            review_scores.append(stars[i])
            if num_word_in_dic < REVIEW_LEN_LIMIT:  # Padding Zero Vecs
                empty = []
                for _ in range(REVIEW_LEN_LIMIT - num_word_in_dic):
                    empty.append(np.zeros(vec_len))
                words_matrix = empty + words_matrix
            words_matrix = np.vstack(words_matrix)
            np.save('./review_matrices/' + str(i_eff_review), words_matrix)

        i += 1
        if not not_found:
            break

    return i

offsets = [i * num_reviews_for_a_star for i in range(5)]
for i in range(REVIEW_NUM):
    if (i+1)%1000 == 0:
        print("word2vec-ing review %d" % (i+1))
    for j in range(5):
        offsets[j] = get_vecs(offsets[j])

review_scores = np.array(review_scores)
for i in range(len(review_scores)):
    print(i+1, ' ', review_selected_list[i], ' ', review_scores[i])
np.save('./review_matrices/review_scores', review_scores)