import streamlit as st
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('all-MiniLM-L6-v2')
import pandas as pd

# my_data = genfromtxt('embed2.csv', delimiter=',')

# df = pd.read_csv('C:/Users/kiran/Python_3/quora-question-pairs_2021/train.csv/train_for_similarity.txt', delimiter = "\t",header=None)
# df = df[0:25000]
df = pd.read_csv('dataset.txt', delimiter='\t', header=None)
sentences = [sentence.replace('br','')
             .replace('<',"")
             .replace(">","")
             .replace('\\',"")
             .replace('\/',"")
             for sentence in df[0]]
embedding = model.encode(sentences)


st.set_page_config(page_title="Sentence Similarity",page_icon="185976.jpg",layout="centered",initial_sidebar_state="expanded")


def preprocess(review):  

    sentence_embeddings1 = model.encode(review)
    l=[]
    cos_sim = util.cos_sim(sentence_embeddings1, embedding)

    winners = []
    for arr in cos_sim:
        for i, each_val in enumerate(arr):
            winners.append([sentences[i],each_val])

    # lets get the top 2 sentences
    final_winners = sorted(winners, key=lambda x: x[1], reverse=True)



    for arr in final_winners[0:10]:
        l.append(arr[0])
    return l
    # sim_arr=cosine_similarity(
    # [sentence_embeddings1],
    # my_data[:]
    # )
    # di=dict(zip(sim_arr[0],df[0]))

    # sim_arr=np.sort(sim_arr[0])[::-1]
    # sim_arr=sim_arr[0:10]

    # for i in sim_arr:
    #     l.append(di[i])

    # return l

    

       
    # front end elements of the web page 
html_temp = """ 
    <div id=1 style ="background-color:orange;padding:10px"> 
    <h1 style ="color:blue;text-align:center;">Search Similar Queries</h1> 
    </div>
    
    <style>
    body {
    background-image: url("https://static.vecteezy.com/system/resources/thumbnails/001/874/132/small/abstract-geometric-white-background-free-vector.jpg");
    background-size: cover;
    }
    </style>
    """

html1 = """<h2 style ="color:blue;text-align:right;"> - DEMO</h2> """
      
# display the front end aspect
st.markdown(html_temp, unsafe_allow_html = True) 
st.markdown(html1, unsafe_allow_html = True)
st.markdown("""<h2 style ="color:blue;text-align:left;">Enter your query below</h2> """, unsafe_allow_html = True)
      
# following lines create boxes in which user can enter data required to make prediction
review=st.text_area("")



#user_input=preprocess(sex,cp,exang, fbs, slope, thal )
pred=preprocess(review)
# pred = ['adf', 'dfg', 'gsf']

if st.button("Search"):    
  for sent in pred:
      st.success(sent)
    

st.sidebar.subheader("About App")

st.sidebar.info("This web app is to search similar queries for the given query")
st.sidebar.info("Enter the required fields and click on the 'Search' button to check the same")
st.sidebar.info("Don't forget to rate this app")

feedback = st.sidebar.slider('How much would you rate this app?',min_value=0,max_value=5,step=1)

if feedback:
  st.header("Thank you for rating the app!")
