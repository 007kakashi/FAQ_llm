from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.llms import GooglePalm
from langchain.document_loaders import CSVLoader
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import pandas as pd
import os

load_dotenv()

faiss_path = os.path.abspath("C:\\ML Projects\\LLM - Projects\\FAQ_llm\\faiss_index")

api = os.environ['GOOGLE_API_KEY']

llm = GooglePalm(google_api_key= api, temperature=0.8)

embedding = HuggingFaceInstructEmbeddings()


def create_vector_db():
        
        try:
        
            loader = CSVLoader(file_path='codebasics_faqs.csv', source_column='prompt')

            data = loader.load()

            vectorstore = FAISS.from_documents(documents= data, embedding= embedding)

            vectorstore.save_local(faiss_path)
        
        except Exception as e:
            print(f"An error occurred while saving the vectorstore: {e}")


def get_qa_chain():
        
        faiss = FAISS.load_local(faiss_path,embeddings= embedding)

        retriever = faiss.as_retriever(score_threshold=0.7)

        prompt_template = """Given the following context and a question, generate an answer based on this context only.
        In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
        If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

        CONTEXT: {context}

        QUESTION: {question}"""

        prompt = PromptTemplate(input_variables=['context', 'question'], template= prompt_template)

        qa = RetrievalQA.from_chain_type(llm, 
                                 retriever= retriever, 
                                 input_key= "query", 
                                 return_source_documents= True,
                                 chain_type_kwargs={"prompt":prompt})
        
        return qa


if __name__ == "__main__":
        create_vector_db()
        # q = get_qa_chain()
        # print(q("what is the javascript course fees").get('result'))


        


