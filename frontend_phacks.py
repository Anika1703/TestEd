import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import matplotlib.pyplot as plt
from io import BytesIO
import numpy as np
import itertools
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import speech_recognition as sr
import matplotlib as mpl
from streamlit_lottie import st_lottie
import requests
from APIfunctions import *


def transcribe_audio(audio_file):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio = recognizer.record(source)
    return recognizer.recognize_google(audio)


# Importing diversification functions
def max_sum_sim(doc_embedding, candidate_embeddings, candidates, top_n, nr_candidates):
    # Calculate distances and extract keywords
    distances = cosine_similarity(doc_embedding, candidate_embeddings)
    distances_candidates = cosine_similarity(candidate_embeddings, candidate_embeddings)

    # Get top_n words as candidates based on cosine similarity
    words_idx = list(distances.argsort()[0][-nr_candidates:])
    words_vals = [candidates[index] for index in words_idx]
    distances_candidates = distances_candidates[np.ix_(words_idx, words_idx)]

    # Calculate the combination of words that are the least similar to each other
    min_sim = np.inf
    candidate = None
    for combination in itertools.combinations(range(len(words_idx)), top_n):
        sim = sum([distances_candidates[i][j] for i in combination for j in combination if i != j])
        if sim < min_sim:
            candidate = combination
            min_sim = sim

    return [words_vals[idx] for idx in candidate]

def mmr(doc_embedding, candidate_embeddings, candidates, top_n, diversity):

    # Extract similarity within words, and between words and the document
    word_doc_similarity = cosine_similarity(candidate_embeddings, doc_embedding)
    word_similarity = cosine_similarity(candidate_embeddings)

    # Initialize candidates and already choose best keyword/keyphrase
    keywords_idx = [np.argmax(word_doc_similarity)]
    candidates_idx = [i for i in range(len(candidates)) if i != keywords_idx[0]]

    for _ in range(top_n - 1):
        # Extract similarities within candidates and
        # between candidates and selected keywords/phrases
        candidate_similarities = word_doc_similarity[candidates_idx, :]
        target_similarities = np.max(word_similarity[candidates_idx][:, keywords_idx], axis=1)

        # Calculate MMR
        mmr = (1-diversity) * candidate_similarities - diversity * target_similarities.reshape(-1, 1)
        mmr_idx = candidates_idx[np.argmax(mmr)]

        # Update keywords & candidates
        keywords_idx.append(mmr_idx)
        candidates_idx.remove(mmr_idx)

    return [candidates[idx] for idx in keywords_idx]


# Define a function to create a mindmap and display it in Streamlit
def create_mindmap(doc, keywords):
    
    # Set up a graph
    G = nx.Graph()
    
    # Add the document as the root node of the graph
    G.add_node(doc, color='blue', size=3000)
    
    # Add each keyword as a child node of the root node
    for keyword in keywords:
        G.add_node(keyword, color='red', size=1000)
        G.add_edge(doc, keyword, weight=0.3)

    # Set the position of the root node
    pos = {doc: (0, 0)}

    # Set the position of each child node using a circular layout
    for i, keyword in enumerate(keywords):
        angle = 2 * i * 3.14 / len(keywords)
        x = 2000 * np.cos(angle)
        y = 2000 * np.sin(angle)
        pos[keyword] = (x, y)

    # Set node colors and sizes
    node_colors = [G.nodes[n]['color'] for n in G.nodes()]
    node_sizes = [G.nodes[n]['size'] for n in G.nodes()]

    # Draw the graph and save it to a buffer
    plt.figure(figsize=(16, 16))
    nx.draw(G, pos, node_color=node_colors, node_size=node_sizes, width=2, font_size=20, with_labels=True)
    plt.axis('off')
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    
    # Display the image in Streamlit
    st.image(buffer)

def sentiment_scores(text):
 
    # Create a SentimentIntensityAnalyzer object.
    sid_obj = SentimentIntensityAnalyzer()
 
    # polarity_scores method of SentimentIntensityAnalyzer
    # object gives a sentiment dictionary.
    # which contains pos, neg, neu, and compound scores.
    sentiment_dict = sid_obj.polarity_scores(text)
    st.write('**We analyzed your response-- here are a couple of pointers on things you did well and things that could use improvement:**')
    st.write("🎓 The tone of your response was ", sentiment_dict['neg']*100, "% Negative")
    st.write("🎓 The tone of your response was ", sentiment_dict['neu']*100, "% Neutral")
    st.write("🎓 The tone of your response was ", sentiment_dict['pos']*100, "% Positive")
 
    st.write('**_The tone of your response was overall..._**', end = " ")
 
    # decide sentiment as positive, negative and neutral
    if sentiment_dict['compound'] >= 0.05 :
        st.write("Positive!")
 
    elif sentiment_dict['compound'] <= - 0.05 :
        st.write("Negative. :(")
 
    else :
       st.write("Neutral!")


def passive(text):
    passive_words = ["am", "is", "been", "was", "are", "be", "being"]
    non_passive_words = ['do', 'did', 'does', 'have', 'has', 'had']
    passive_counter = 0 
    non_passive_counter = 0 
    broken = text.split()
    for i in broken:
        if i in passive_words: 
            passive_counter += 1
        elif i in non_passive_words: 
            non_passive_counter += 1 
    if non_passive_counter!= 0:
        passive_ratio = passive_counter/non_passive_counter
        return passive_ratio
    else: 
        return 0

def confidence(text):
    not_confident = ["like", "umm", "um", "but", "uh", "you know", "i mean", "so", "just", "basically", "i guess"]
    weakling_counter = 5
    broken = text.split()
    for i in broken: 
        if i in not_confident:
            weakling_counter -= 1
    return weakling_counter

def get_formality_score(text):
    formal_words = ["please", "kindly", "thank you", "sincerely", "yours truly"]
    informal_words = ["gonna", "wanna", "ain't", "don't", "can't"]
    formal_counter = 0
    informal_counter = 0
    formality_ratio = 0

    broken = text.split()
    for i in broken:
        if i in formal_words:
            formal_counter += 1
        elif i in informal_words:
            informal_counter -= 1
    if informal_counter!= 0: 
        formality_ratio = formal_counter/informal_counter
    return formality_ratio


def suggestions(text):

    if passive(text) > 1: 
        st.write("✔️ Your tone is :violet[**too passive**]. Using passive words makes the overall tone of your response less engaging-- to make that awesome stream of conciousness ✨shine✨ from within, **use active words** like 'do', 'does' and 'have'. ")
    
    if passive(text) < 1: 
        st.write("✔️ Your response uses :orange[non-passive words] for the most part! Good job!")
    
    if confidence(text) <= 0: 
        st.write("✔️ :violet[**Your tone could be more confident.**] Avoid using filler words and take pauses instead of rambling on with the 'ums' and 'buts'. Remember that you are awesome and you got this! 🙌 ")

    if confidence(text) > 0: 
        st.write("✔️ Wow, you sound :orange[confident]! Keep it up! 😎")
    
    if get_formality_score(text) > 1:
        st.write("✔️ The overall tone of your response is :red[**formal**].")
    
    if get_formality_score(text) < 1: 
        st.write("✔️ The overall tone of your response is :red[**informal**].")
    
    if get_formality_score(text) == 0: 
        st.write("✔️ The overall tone of your response in terms of formality is :red[**neutral!**]")
    


# # Take input text from the user
# doc = st.text_area("Enter the text to extract keywords from:")

# if doc.strip():
#     # Extract candidate words/phrases
#     n_gram_range = (3, 3)
#     stop_words = "english"
#     count = CountVectorizer(ngram_range=n_gram_range, stop_words=stop_words).fit([doc])
#     candidates = list(count.vocabulary_.keys())

#     # Encode text and candidate keywords
#     model = SentenceTransformer('distilbert-base-nli-mean-tokens')
#     doc_embedding = model.encode([doc])
#     candidate_embeddings = model.encode(candidates)

#     # Calculate cosine similarity between text and candidate keywords
#     top_n = 5
#     nr_candidates = 20
#     diversity = 0.5

#     # Use max_sum_sim to diversify keywords
#     keywords = max_sum_sim(doc_embedding, candidate_embeddings, candidates, top_n, nr_candidates)
#     # Alternatively, use mmr to diversify keywords
#     # keywords = mmr(doc_embedding, candidate_embeddings, candidates, top_n, diversity)

#     # Display top keywords
#     st.write("Top Keywords:")
#     for keyword in keywords:
#         st.write(keyword)
    
#     overall = sentiment_scores(doc)
#     more_suggestions = suggestions(doc)
#     st.write(more_suggestions)
#     # Create a mindmap and display it
#     create_mindmap(doc, keywords)
    
# else:
#     st.write("Please enter some text")

def welcome_page():
    st.title("Welcome to TestEd.")
    st.write("The one stop shop for revision.")
    st.write("Click on the arrow below to get started.")
    if st.button("▼"):
        st.session_state["page"] = "tool_selection"


def tool_selection_page():
    st.title("Select a tool")
    st.write("Choose a tool from the list below and click 'Next' to continue.")
    tool = st.selectbox("", ["Mindmapper Tool", "Transcriber & Summarizer", "Mock Test"], index=0)
    st.write(get_tool_description(tool))
    if st.button("Next"):
        st.session_state["page"] = "upload_file"
        st.session_state["tool"] = tool

def question_page(questions):
    st.title("Answer the questions below")
    # Take input text from the user
    st.success(questions)
    st.text_area("Enter your answer below:")


def upload_file_page():
    global x
    global y
    st.title(f"{st.session_state['tool']} - Upload a file")
    if st.session_state['tool'] == "Mindmapper Tool":
        uploaded_file = st.file_uploader("Choose a file", type=["pdf", "mp3"])
        if uploaded_file is not None:
            st.success("File uploaded successfully")
            if uploaded_file.type.split('/')[1] == 'pdf':
                x = CreateMindmap(TranscriptSummarize(ReadPDF(uploaded_file.name)))
                for i in x:
                    st.write(i)
            else:
                x = CreateMindmap(TranscriptSummarize(AudioToText(uploaded_file.name)))
                for i in x:
                    st.write(i)

    elif st.session_state['tool'] == "Transcriber & Summarizer":
        uploaded_file = st.file_uploader("Choose a file", type=["pdf", "mp3"])
        if uploaded_file is not None:
            st.success("File uploaded successfully")
            if uploaded_file.type.split('/')[1] == 'pdf':
                x = TranscriptSummarize(ReadPDF(uploaded_file.name))
                st.write("Summary:")
                st.write(x)
            else: 
                x = AudioToText(uploaded_file.name)
                y = TranscriptSummarize(x)
                st.write("Transcript:")
                st.write(x)
                st.write("Summary:")
                st.write(y)

    else:
        uploaded_file = st.file_uploader("Choose a file", type=["pdf", "mp3"])
        if uploaded_file is not None:
            st.success("File uploaded successfully")
            if uploaded_file.type.split('/')[1] == 'pdf':
                x,y = TestGeneration(TranscriptSummarize(ReadPDF(uploaded_file.name)))
                st.write("Questions:")
                st.write(x)
                doc = st.text_area("Enter your answer below:")

                if doc.strip():
                    # Extract candidate words/phrases
                    n_gram_range = (3, 3)
                    stop_words = "english"
                    count = CountVectorizer(ngram_range=n_gram_range, stop_words=stop_words).fit([doc])
                    candidates = list(count.vocabulary_.keys())

                    # Encode text and candidate keywords
                    model = SentenceTransformer('distilbert-base-nli-mean-tokens')
                    doc_embedding = model.encode([doc])
                    candidate_embeddings = model.encode(candidates)

                    # Calculate cosine similarity between text and candidate keywords
                    top_n = 5
                    nr_candidates = 20
                    diversity = 0.5

                    # Use max_sum_sim to diversify keywords
                    keywords = max_sum_sim(doc_embedding, candidate_embeddings, candidates, top_n, nr_candidates)
                    # Alternatively, use mmr to diversify keywords
                    # keywords = mmr(doc_embedding, candidate_embeddings, candidates, top_n, diversity)

                    # Display top keywords
                    # st.write("Top Keywords:")
                    # for keyword in keywords:
                    #     st.write(keyword)
                    
                    overall = sentiment_scores(doc)
                    more_suggestions = suggestions(doc)
                    st.write(more_suggestions)
                    # Create a mindmap and display it
                    # create_mindmap(doc, keywords)

                    if st.button("Show Answers"):
                        st.write(y)
                

                
            else:
                x,y = TestGeneration(TranscriptSummarize(AudioToText(uploaded_file.name)))
                st.write("Questions:")
                st.write(x)
                doc = st.text_area("Enter your answer below:")
                
                if doc.strip():
                    # Extract candidate words/phrases
                    n_gram_range = (3, 3)
                    stop_words = "english"
                    count = CountVectorizer(ngram_range=n_gram_range, stop_words=stop_words).fit([doc])
                    candidates = list(count.vocabulary_.keys())

                    # Encode text and candidate keywords
                    model = SentenceTransformer('distilbert-base-nli-mean-tokens')
                    doc_embedding = model.encode([doc])
                    candidate_embeddings = model.encode(candidates)

                    # Calculate cosine similarity between text and candidate keywords
                    top_n = 5
                    nr_candidates = 20
                    diversity = 0.5

                    # Use max_sum_sim to diversify keywords
                    keywords = max_sum_sim(doc_embedding, candidate_embeddings, candidates, top_n, nr_candidates)
                    # Alternatively, use mmr to diversify keywords
                    # keywords = mmr(doc_embedding, candidate_embeddings, candidates, top_n, diversity)

                    # Display top keywords
                    # st.write("Top Keywords:")
                    # for keyword in keywords:
                    #     st.write(keyword)
                    
                    overall = sentiment_scores(doc)
                    more_suggestions = suggestions(doc)
                    st.write(more_suggestions)
                    # Create a mindmap and display it
                    # create_mindmap(doc, keywords)

                    if st.button("Show Answers"):
                        st.write(y)
                    
                # else:
                #     st.write("Please enter some text")
    

def get_tool_description(tool):
    if tool == "Mindmapper Tool":
        return "The mindmapper tool allows you to convert your lecture notes and lecture recordings into a visual mindmap to make your revision process for your next exam more efficient."
    elif tool == "Transcriber & Summarizer":
        return "The summarizer tool gives you an easily understandable and readable summary of the transcript of a lecture recording providing you with all the key points mentioned during class."
    elif tool == "Mock Test":
        return "The mock test tool generates a sample exam paper with questions and answers on the basis of your lecture notes and lecture recording."


def main():
    st.set_page_config(page_title="TestEd", page_icon="pencil.png")

    lottie_url = "https://assets9.lottiefiles.com/packages/lf20_Kcued0rJrz.json"
    lottie_json = requests.get(lottie_url).json()

    st_lottie(lottie_json, speed=1, width=200, height=200, key="animation")

    if "page" not in st.session_state:
        st.session_state["page"] = "welcome"

    if st.session_state["page"] == "welcome":
        welcome_page()
    elif st.session_state["page"] == "tool_selection":
        tool_selection_page()
    elif st.session_state["page"] == "upload_file":
        upload_file_page()
    elif st.session_state["page"] == "question_page":
        question_page()


if __name__ == "__main__":
    main()   