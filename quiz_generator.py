from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="text-generation",
    max_new_tokens=800
)

model = ChatHuggingFace(llm=llm)
parser = StrOutputParser()

prompt1 = PromptTemplate(
    template="Generate a short note for the following text. \n {text}",
    input_variables=["text"]
)

prompt2 = PromptTemplate(
    template="Generate a quiz with 5 question and answers for the follwing text. \n {text}",
    input_variables=["text"]
)

prompt3 = PromptTemplate(
    template="Merge the following note and quiz into a single output. \n Note -> {note} \n Quiz -> {quiz}",
    input_variables=["note", "quiz"]
)

parallel_chain = RunnableParallel({
    "note": prompt1 | model | parser,
    "quiz": prompt2 | model | parser
})

merge_chain = prompt3 | model | parser

chain = parallel_chain | merge_chain

text = """
ChatGPT is an advanced artificial intelligence assistant developed by OpenAI to help people interact with technology through natural language. Built on large language models, ChatGPT is capable of understanding context, interpreting user requests, and generating human-like responses across a wide range of topics and domains.

Its primary purpose is to assist users with information retrieval, problem-solving, learning, content creation, software development, research, and decision-making. ChatGPT can explain complex concepts in simple terms, provide detailed technical guidance, generate creative content, summarize large documents, analyze data, and support educational and professional tasks. It is widely used by students, researchers, software engineers, business professionals, and content creators.

One of ChatGPT's key strengths is its adaptability. It can tailor its responses based on the user's level of expertise, whether explaining basic programming concepts to a beginner or discussing advanced machine learning techniques with an experienced developer. It can communicate in multiple languages and assist with tasks such as translation, proofreading, brainstorming, report writing, and coding.

In the field of software development, ChatGPT can generate code in various programming languages, identify bugs, suggest optimizations, explain algorithms, and help developers understand frameworks and libraries. It can also assist with database design, cloud computing concepts, system architecture, and software engineering best practices.

Beyond technical domains, ChatGPT can contribute to creative endeavors by generating stories, articles, poems, marketing content, social media posts, and business documents. It can help organize ideas, refine writing, and provide constructive suggestions for improving content quality and clarity.

Despite its extensive capabilities, ChatGPT does not possess consciousness, emotions, beliefs, or personal experiences. It does not think or reason in the same way humans do. Instead, it generates responses by identifying patterns in the vast amounts of text data on which it was trained. As a result, while it often provides useful and accurate information, it can occasionally make mistakes or produce outdated information, particularly regarding recent events if it does not have access to current data sources.

ChatGPT is designed with a focus on safety, usefulness, and accessibility. It aims to provide helpful responses while adhering to guidelines that promote responsible and ethical use of artificial intelligence. Through continuous improvements and updates, ChatGPT strives to become a more reliable, knowledgeable, and versatile assistant capable of supporting users in both everyday tasks and specialized professional activities.

Today, ChatGPT serves millions of users worldwide as a conversational AI system that bridges the gap between human language and computational intelligence, enabling more intuitive and productive interactions with technology.
"""

result = chain.invoke({"text": text})

print(result)