## Integrate code with OpenAI
import os
from constants import openai_key
from langchain.llms import OpenAI, openai
from langchain_community.llms import OpenAI
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SimpleSequentialChain

# Set the open ai secret key
os.environ["OPENAI_API_KEY"] = openai_key

# Streamlit framework
st.title("Search about your Celebrity")
input_text = st.text_input("Search the topic you want")

# Prompt Template
first_input_prompt = PromptTemplate(
    input_variables= ['name'],
    template="Tell me about celebrity {name}"
)

## OpenAI LLMs
llm = openai.OpenAI(temperature=0.8)
chain = LLMChain(llm=llm,
         prompt=first_input_prompt,
         verbose=True,
         output_key="person")



# Prompt Template
second_input_prompt = PromptTemplate(
    input_variables= ['person'],
    template="When was  {person} born"
)

chain2 = LLMChain(llm=llm,
         prompt=second_input_prompt,
         verbose=True,
         output_key="dob")


parent_chain = SimpleSequentialChain(chains=[chain, chain2],
                      verbose=True)

if input_text:
    st.write(parent_chain.run(input_text))