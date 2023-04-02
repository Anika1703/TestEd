import openai
import os
import requests
import json
import AudioToTextUtils
import pdfplumber
from summa.summarizer import summarize

def AudioToText(AudioFile):
    upload_endpoint = "https://api.assemblyai.com/v2/upload"
    transcript_endpoint = "https://api.assemblyai.com/v2/transcript"

    api_key = "3e58f76fabbb4668b50c1da8bb44ff80"

    if api_key is None:
        raise RuntimeError ("You have not added your API key to the program")

    Author = {'authorization': api_key, 'content-type': 'application/json'}

    upload_url = requests.post(upload_endpoint, headers=Author, data=AudioToTextUtils.readfile(AudioFile)).json()

    transcript_request = {'audio_url': upload_url['upload_url']}
    transcript_response = requests.post(transcript_endpoint, json=transcript_request, headers=Author).json()
    
    polling_endpoint = "https://api.assemblyai.com/v2/transcript/"+transcript_response['id']

    AudioToTextUtils.WaitForCompletion(polling_endpoint, Author)

    paragraphs_response = requests.get(polling_endpoint+"/paragraphs", headers=Author).json()

    paragraphs = []
    for para in paragraphs_response['paragraphs']:
        paragraphs.append(para['text'] + '\n')

    return ''.join(paragraphs)

def TranscriptSummarize(Transcript):
    summary = summarize(Transcript, ratio = 0.05)
    summary = summary.split('\n')
    return '\n'.join(summary[0:-1])

def ReadPDF(filename):
    with open(filename, 'rb') as pdf_file:
        list1 = []
        pdf = pdfplumber.open(pdf_file)
        for page in pdf.pages:
            text = page.extract_text()
            list1.append(text)

    return ''.join(list1)

def TestGeneration(summary):
    openai.api_key = "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"

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that can create test papers from a given text."},
            {"role": "user", "content": "The Test paper should consist of only questions and absolutely no other text."},
            {"role": "assistant", "content": "Alright, I will not write any other text when giving the paper."},
            {"role": "user", "content": "The paper should only contain short answer questions with a maximum of 5 questions."},
            {"role": "assistant", "content": "Understood only 5 questions having answers that are 1 or 2 lines."},
            {"role": "user", "content": "In clustering, basically you recall items better if you can recognize that there are two or more types of things in a particular list. Then when you’re trying to recall the items during the exam, you begin your walk around the house. The better place to try and place all the information you want to learn is in the location where you’re going to have to recall the material. Then as we see in slide 15, when you’re given a list of words to recall, you associate the first word in the list with the peg word. For example you have a word, let’s say you’re trying to recall the word “Bee” and the peg word is bun. Not within the middle of the list that you’re trying to look at you only have about a 20% chance of recalling a particular set of items. Let’s give an example, if you learn list A, then you learn list B, and finally you have to recall B. Now there’s a related concept that goes with the similarity of learning and recall condition and this is shown in slide 35. “It’s best to recall information in the same drug state as you are when you’re doing the learning.” So, as we see on slide 36, if you smoke, you need to smoke while taking your exam."},
            {"role": "assistant", "content": "1. What is overlearning, and how can it be useful for studying? 2. How can organization improve memory, and what is clustering? 3. What are some other organizational strategies besides clustering that can improve memory? 4. How can the similarity between the learning and recall conditions affect memory, and what is an example of this concept? 5. What is the related concept to the similarity between the learning and recall conditions, and what is an example of this concept?"},
            {"role": "user", "content": f"Okay now create a test on this text: {summary}"}
            ]
        )
    QuestionPaper = response['choices'][0]['message']['content']

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that can give me answers for my questions."},
            {"role": "user", "content": "The response should consist of only answers and absolutely no other text."},
            {"role": "assistant", "content": "Alright, I will not write any other text when giving the response."},
            {"role": "user", "content": "The answers should only be a maximum of 1 or 2 lines"},
            {"role": "assistant", "content": "Understood answers to each questions with only 1 or 2 lines."},
            {"role": "user", "content": "1. What is overlearning, and how can it be useful for studying? 2. How can organization improve memory, and what is clustering? 3. What are some other organizational strategies besides clustering that can improve memory? 4. How can the similarity between the learning and recall conditions affect memory, and what is an example of this concept? 5. What is the related concept to the similarity between the learning and recall conditions, and what is an example of this concept?"},
            {"role": "assistant", "content": "1. Overlearning is studying something after it can be recalled perfectly, which can help solidify information in memory and make comprehensive exams easier. 2. Organization can improve memory by making it easier to recall information. Clustering is grouping items based on similarities. 3. Other organizational strategies include categorization, sequencing, and elaboration. 4. Similarity between learning and recall conditions can improve memory. For example, studying in the same environment where you will be taking the test can enhance recall. 5. The related concept is context-dependent memory, which refers to the idea that recall is improved when the context of learning and recall are similar. For example, if you learned information while sitting in a particular chair, you may recall that information better if you are sitting in that same chair during recall."},
            {"role": "user", "content": f"Okay now answer the following questions: {QuestionPaper}"}
            ]
        )

    AnswerSheet = response['choices'][0]['message']['content']
    
    return QuestionPaper, AnswerSheet

def CreateMindmap(summary):
    openai.api_key = "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"

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that can create a mind map on the main points of a given text."},
            {"role": "user", "content": "It should be a simple mind map with one main topic and subheadings based on the content"},
            {"role": "assistant", "content": "Alright, I will produce a simple mindmap for you"},
            {"role": "user", "content": "Create a mind map to study the content of the following text: In the last section, we examined some early aspects of memory. In this section, what we’re going to do is discuss some factors that influence memory. So let’s do that by beginning with the concept on slide two, and that concept is overlearning. Basically in overlearning, the idea is that you continue to study something after you can recall it perfectly. So you study some particular topic whatever that topic is. When you can recall it perfectly, you continue to study it. This is a classic way to help when one is taking comprehensive finals later in the semester. So when you study for exam one and after you really know it all, you continue to study it. That will make your comprehensive final easier. The next factor that will influence memory relates to what we call organization. In general, if you can organize material, you can recall it better. There are lots of different types of organizational strategies and I’ve listed those on slide four. So let’s begin by talking about the first organizational strategy called clustering and is located on page five. In clustering, basically you recall items better if you can recognize that there are two or more types of things in a particular list. So let’s give a couple of lists and show you some examples of that. These examples are shown in slide six. "},
            {"role": "assistant", "content": """Factors that Influence Memory
|
|----> Overlearning
| |----> Continuously studying after perfect recall
| |----> Helpful for comprehensive exams
|
|----> Organization
| |----> Improves recall
| |----> Various organizational strategies
| |----> Clustering
| |----> Recognize two or more types of things in a list"""},
            {"role": "user", "content": f"Okay now create a to study the content of the following text: {summary}"}
            ]
        )
    Map = response['choices'][0]['message']['content']
    Map = Map.split("\n")

    return Map