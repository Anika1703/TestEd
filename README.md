# TestEd.

![tested](https://user-images.githubusercontent.com/63713718/229340189-276310b0-7d0d-44da-b432-8cea75b9ad9c.png)

The all-in-one educational tool to revolutionize your study game with AI-generated mindmaps, summaries, mock tests, and flashcards. Make revision easy and efficient. Get TestEd today (pun intended).

## Inspiration
In the US, there are over 6.1 million school-going children diagnosed with ADHD. For students with ADHD, acing a midterm isn’t as simple as grinding through a study session. It can often be daunting, frustrating, and tiring, especially when the education system is designed to be normalized to the needs of the more able. So, when brainstorming our project, the primary goal was not to help students with ADHD study longer or harder but to help them study **differently** – in ways that suit **their** learning needs. 

This problem is also personal to our team. Two of our members have ADHD and belong to communities where it isn’t widely accepted. Since one of the biggest problems associated with ADHD is inattentiveness and distractibility, there exists the need for a tool that can help minimize the impact of these on long-term academic success. 

That's where we come in! 

## What it does
TestEd serves as a one-stop platform to seamlessly transition from one way of learning to another, depending on your needs. Here is how our tool deals with different problems encountered by students with ADHD.
- Reading: Chugging through a long, never-ending text for your APUSH exam can seem daunting. Reading is a passive activity for the ADHD brain. Our test generator tool takes in audio or text inputs and generates practice questions to help you stay engaged. TestEd even provides feedback through sentiment analysis, suggesting ways to improve your responses. So for example, if your response has too many filler words, like ‘but’, TestEd will comment on it and suggest ways to circumvent that shortcoming. 
- Listening: There is nothing worse than being unable to focus on a 1.5-hour lecture on a topic you're actually interested in. To help ADHD brains deal with this problem, we have a summary generator feature that summarizes the main points of a lecture for you. You can either upload a transcript of your lecture or an mp3/wav file. The summary generator feature simplifies lengthy lectures by extracting key points, allowing you to stay focused and retain information.
- Visualizing: Our mindmap tool converts lecture notes and recordings into visual maps, making revision more efficient.

## How we built it
Our tech stack for TestEd included:
- Streamlit for the frontend and backend integration
- Python as the backbone for frontend and database construction
- vaderSentiment for sentiment analysis
- summa for text summarization
- BERT for extracting key phrases from text
- PDFPlumber to read pdf files
- json to interact with the API
- AssemblyAI's API for speech-to-text conversion
- OpenAI's API for mock test and mindmap generation
- Figma for prototyping the Phase 2 development

##Team Members (Discord Usernames):
- Arnav Nigam: @GodMagdon117#6014
- Anika Sharma: @Anika17#2415
- Teena Bhatia: @Teena#9111
- Siddhartha Reddy Pullannagari: @sidzz#3594

