import json
import pandas as pd

#stars = []
#reviews = []
reviews_len = []
#num_lines = sum(1 for line in open('not_editable/review.json'))
#num_lines 4736897
REVIEW_NUM = 4736896
with open("../not_editable/review.json") as myfile:
    for x in xrange(REVIEW_NUM):
        text = json.loads(next(myfile))
        l = len(text['text'].split(' '))
        reviews_len.append(l)
        
myfile.close()    

#df = pd.DataFrame(data={'stars': stars, 'reviews': reviews})
#df.to_excel('reviews_rating300limit.xlsx', sheet_name='sheet1', index=False)
#df.to_pickle('reviews_rating300limit_.pkl')  # where to save it, usually as a .pkl
#df1 = pd.read_pickle('reviews_rating300limit.pkl')
#4451334