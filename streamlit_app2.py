
import streamlit as st 
import langchain 
import chromadb 
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())
import os 

class StreamlitApp:
    def __init__(self):
        st.title('Interfaz gráfica TFG Elena Abad')
        self.openai_api_key = st.sidebar.text_input('OpenAI API Key', type='password')

        # Ruta al directorio de documentos
        self.docs_dir = st.sidebar.text_input('Path to documents directory', value='./docs')

        # Asegurémonos de que la ruta sea absoluta
        self.docs_dir = os.path.abspath(self.docs_dir)

    def generate_response(self, input_text):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=400)
        all_splits = text_splitter.split_documents(input_text)

        vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        llm = ChatOpenAI(temperature=0.7, model="gpt-3.5-turbo")
        qa = ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever(), memory=memory)

        response = qa.process(input_text=input_text)
        return response

    def run(self):
        with st.form('my_form'):
            text = st.text_area('Introduzca una instrucción', 'Una persona tiene un salario neto de 2000 euros al mes. Solicita una hipoteca por importe de 400000 euros. La tasa de riesgo hipotecario que prescriben entidades como el Banco de España es de un 30 por ciento del salario mensual.¿Cuánto tiene que pagar al mes?"')
            submitted = st.form_submit_button('Enviar')

            if not self.openai_api_key.startswith('sk-'):
                st.warning('Por favor, introduzca su clave de OpenAI (OpenAI API Key).', icon='⚠')

            if submitted and self.openai_api_key.startswith('sk-'):
                response = self.generate_response(text)
                st.info(response)

if __name__ == "__main__":
    app = StreamlitApp()
    app.run()