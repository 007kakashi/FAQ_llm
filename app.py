import streamlit as st
from helper import get_qa_chain
st.header('ChatBot Assistant')

input = st.text_input("What is Your Concern?")

ask = st.button('Ask')

if ask:
    q = get_qa_chain(input)
    st.write(q)