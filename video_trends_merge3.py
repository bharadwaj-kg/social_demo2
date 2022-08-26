
#!pip install librosa


# In[ ]:


import streamlit as st
import os
import json

#--------------------------------------------------------------------------------
#selectbox1 = st.sidebar.selectbox("Keyword Based Trends"),["Trends Insight","Trends Description","Brand Mentions","Composition/Ingredients"])
#selectbox2 = st.sidebar.selectbox("Video Based Trends",["Trends Insight","Trends Description", "Trends of Each Influencers"])
#st.set_page_config(layout="centered")
st.write("""
         # Video Based Trend Detection
         """
         )
#st.write("Speech to text conversion")
header = st.container()

#--------------------------------------------------------------------------------
#video = st.container()
#if selectbox2 == "Trends Insight":
with header:
     st.header("Speech to Text")

file = st.file_uploader("Please upload a video file", type=["mp4"])

if file:

    filename = file.name
    with open(filename, mode='wb') as f:
      f.write(file.read())
      videos = st.container()
      with videos:
           
           _left, mid, _right = st.columns(3) 
           with _left:
                st.header('Video Display')
                st.video(filename)
           #st.video(filename)  

    #os.system("ffmpeg -i decisions.mp4 decisions.wav")
    #os.system('ffmpeg -i'+' '+ str(filename) +' '+ str(os.path.splitext(str(filename))[0])+'.wav')
    #wavname = str(os.path.splitext(str(filename))[0])+'.wav'
    
    
# In[2]:


#import nltk
    #import librosa
    #import torch
    #import soundfile as sf
    from transformers import pipeline
    from transformers import Wav2Vec2ForCTC, Wav2Vec2CTCTokenizer
    #nltk.download('punkt')
    
    model_path = "C:/Users/GovindaBharadwajKoll/w2v_model_new/"
     
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(model_path)
    model = Wav2Vec2ForCTC.from_pretrained(model_path)
    
    pipe = pipeline(task = "automatic-speech-recognition", model= model, tokenizer = tokenizer,
               feature_extractor = model_path)
    
    output = pipe(filename, chunk_length_s=30)#, stride_length_s=(4, 2))
        
    text = output["text"].lower()    
    


    # In[ ]:


    if file is None:
        st.text("Please upload a video file")
    else:
        with _right:
             st.header("Converted Text")
             st.write(text)

    st.header("Sample Videos")
    thumbnails = st.container()
    sample_text = st.container()
    from PIL import Image
    with thumbnails:
         #st.header("Video Thumbnails")
         t1,t2,t3,t4,t5,t6 = st.columns(6)
    with t1:
         h1 = Image.open('eyelift.jpg')
         h1_n = h1.resize((200, 150))
        
         st.image(h1_n)#, use_column_width='always', caption='t1')
    with t2:
         h2 = Image.open('blush_draping.jpg')
         h2_n = h2.resize((200, 150))

         st.image(h2_n)#, use_column_width='always',caption='t2')
    with t3:
         h3 = Image.open('method4.jpg')
         h3_n = h3.resize((200, 150))

         st.image(h3_n)#, use_column_width='always',caption='t3')     
    with t4:
         h4 = Image.open('mascara6.jpg')
         h4_n = h4.resize((200, 150))

         st.image(h4_n)#, use_column_width='always',caption='t4') 
    with t5:
         h5 = Image.open('concealer9.jpg')
         h5_n = h5.resize((200, 150))

         st.image(h5_n)#, use_column_width='always',caption='t5')
    with t6:
         st.write("....")
    
    with sample_text:
         #st.header("Sample Text")
         s1,s2,s3,s4,s5,s6 = st.columns(6)
    with s1:
         with open('v1.txt') as f:
              text = f.read()
              st.write(text)  
    with s2:
         with open('v2.txt') as f:
              text = f.read()
              st.write(text)         
    with s3:
         with open('v3.txt') as f:
              text = f.read()
              st.write(text)
    with s4:
         with open('v4.txt') as f:
              text = f.read()
              st.write(text)
    with s5:
         with open('v5.txt') as f:
              text = f.read()
              st.write(text)
    with s6:
         st.write("....")
#--------------------------------------------------------------------------------
    topic_container = st.container()
    import pickle
    from bertopic import BERTopic
    import pandas as pd
    import re
    import emoji

    import spacy
    nlp_model = spacy.load('en_core_web_md', disable=['parser', 'ner'])

    from sklearn.feature_extraction.text import CountVectorizer
    #--------------------------------------------------------------------------------
    
    # Datasets:
    # data_file_dir = "../../reddit_data/"
    data_file_dir = "./reddit_data/"

    data_filename_list_0 = [
    "prod_reddit_Organic_cleanser.pickle",
    "prod_reddit_Organic_moisturizer.pickle",
    "prod_reddit_Organic_serums.pickle",
    "prod_reddit_Plantbased_cleanser.pickle",
    "prod_reddit_Plantbased_eyecare.pickle",
    "prod_reddit_Plantbased_lipcare.pickle",
    "prod_reddit_Plantbased_moisturizer.pickle",
    "prod_reddit_Plantbased_serums.pickle",
    "prod_reddit_Vegan_cleanser.pickle",
    "prod_reddit_Vegan_foundation.pickle",
    "prod_reddit_Vegan_moisturizer.pickle",
    "prod_reddit_Vegan_moisturizers.pickle",
    "prod_reddit_Vegan_serums.pickle",
    "prod_reddit_Vegan_toner.pickle",
    ]
    data_filename_list = [data_file_dir+data_filename for data_filename in data_filename_list_0]
    #--------------------------------------------------------------------------------
    comments_list = []

    for ff, filename in enumerate(data_filename_list):
        #print(filename)
        #print("---")
        with open(filename, 'rb') as handle: data_dict = pickle.load(handle)
        for comment_idx in range(1, 101):
            comment = " ".join([data_dict[comment_idx][key] for key in ["title", "coments_body"]]) #Note: comment=title+comment_body
            comments_list.append(comment)

    df = pd.DataFrame({'comment': comments_list})
    comments_list = df['comment'].unique().tolist()
    #--------------------------------------------------------------------------------
    # Remove urls:
    commentsList_URL = []
    for text in comments_list:
        #print("http" in text)
        new_text = re.sub(r"http\S+", "", text)
        commentsList_URL.append(new_text)

    # Remove graphic emojis:
    emoji_dict = emoji.UNICODE_EMOJI["en"]

    commentsList_URLEmoji = []
    for text in commentsList_URL:
        tokenized_list = [token.lemma_ for token in nlp_model(text)] #check if lemmatizing here doesn anything at all, bcoz I've lemmatized before as well.
        new_text = " ".join([tok for tok in tokenized_list if tok not in emoji_dict])
        commentsList_URLEmoji.append(new_text)
    
    topic_model = BERTopic.load("my_model")
    #fig = topic_model.visualize_barchart(top_n_topics=10, n_words=10, height=400)
    #--------------------------------------------------------------------------------
    

    with topic_container:
         fig_barchart = topic_model.visualize_barchart(top_n_topics=10, n_words=10, height=400)
         st.plotly_chart(fig_barchart)

         fig_intertopic_dist = topic_model.visualize_topics()
         st.plotly_chart(fig_intertopic_dist)

         fig_viz_docs = topic_model.visualize_documents(commentsList_URLEmoji)
         st.plotly_chart(fig_viz_docs)