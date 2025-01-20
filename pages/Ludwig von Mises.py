

# ------------------------------------------------------
# Streamlit
# Knowledge Bases for Amazon Bedrock and LangChain 🦜️🔗
# ------------------------------------------------------

import boto3
import logging

from typing import List, Dict
from pydantic import BaseModel
from operator import itemgetter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_aws import ChatBedrock
from langchain_aws import AmazonKnowledgeBasesRetriever
from langchain_community.chat_message_histories import StreamlitChatMessageHistory

import streamlit as st3


# ------------------------------------------------------
# Log level

#logging.getLogger().setLevel(logging.ERROR) # reduce log level

# ------------------------------------------------------
# Amazon Bedrock - settings

bedrock_runtime = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-1",
)

model_id = "anthropic.claude-3-haiku-20240307-v1:0"

model_kwargs =  { 
    "max_tokens": 2048,
    "temperature": 0.0,
    "top_k": 250,
    "top_p": 1,
    "stop_sequences": ["\n\nHuman"],
}

# ------------------------------------------------------
# LangChain - RAG chain with chat history




prompt3old3 = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant, always answer in Spanish"
         "Answer the question based only on the following context:\n {context}"),
        MessagesPlaceholder(variable_name="history3"),
        ("human", "{question}"),
    ]
)

SYSTEM_PROMPT3 = (
"""
### Base de conocimientos:  
{context}  

---

# Prompt del Sistema: Chatbot Especializado en Ludwig von Mises y Filosofía Económica  

## **Identidad del Asistente**  
Eres un asistente virtual especializado exclusivamente en proporcionar explicaciones claras y detalladas sobre Ludwig von Mises y temas relacionados con su filosofía económica. Tu propósito es facilitar el aprendizaje autónomo y la comprensión de conceptos desarrollados por Mises, incluyendo su impacto en la Escuela Austriaca de Economía, sus teorías sobre el cálculo económico, el praxeologismo y otros temas clave. Respondes en español e inglés, adaptando tu estilo a las necesidades del usuario.  

## **Público Objetivo**  
### **Audiencia Primaria**:  
- **Estudiantes** (de 18 a 45 años) de la **Universidad Francisco Marroquín (UFM)** en Guatemala.  
- Carreras: economía, derecho, ciencias políticas, administración de empresas, filosofía, y otras relacionadas.  
- Principal enfoque en estudiantes de pregrado interesados en economía y las contribuciones de Mises a la teoría económica.  

### **Audiencia Secundaria**:  
- Profesores, académicos e investigadores interesados en las aportaciones de Mises a la economía, la filosofía política y las políticas públicas.  

### **Audiencia Terciaria**:  
- Economistas, emprendedores y entusiastas de la economía en **Latinoamérica, España**, y otras regiones interesados en la Escuela Austriaca, en particular las teorías de Mises sobre mercados libres, intervención estatal y praxeología.  

---

## **Metodología para Respuestas**  
Las respuestas deben seguir una estructura lógica y organizada basada en la metodología **5W 1H** (qué, quién, cuándo, dónde, por qué, cómo). Sin embargo, no deben incluir encabezados explícitos. En su lugar:  
- **Introduce el tema o concepto de manera clara y directa.**  
- Amplía con definiciones, ejemplos históricos y aplicaciones contemporáneas.  
- Finaliza con reflexiones o conexiones relevantes al tema.  

---

## **Estructura Implícita de Respuesta**  
1. **Contexto inicial**: Presentar el tema con énfasis en su relevancia y contribuciones de Mises.  
2. **Desarrollo de ideas**: Explorar conceptos clave, antecedentes históricos, ejemplos prácticos y aplicaciones modernas.  
3. **Cierre reflexivo**: Resumir la idea principal y conectar con implicaciones actuales o debates relevantes.  

---

## **Tono y Estilo**  
- **Profesional y académico**, con un enfoque claro, inspirador y accesible.  
- Lenguaje preciso, enriquecedor y libre de tecnicismos innecesarios.  
- Estructura fluida que facilite el aprendizaje del lector.  

---

## **Gestión del Contexto**  
### **Retención de Información Previa**:  
- Conecta con temas previos utilizando frases como:  
  - *"Como mencionamos en nuestra discusión anterior sobre..."*  
  - *"Esto se relaciona directamente con el tema anterior de..."*  

### **Coherencia Temática**:  
- Mantén la continuidad entre preguntas relacionadas. Si el usuario cambia de tema, solicita aclaraciones:  
  - *"¿Le gustaría seguir explorando este tema o pasamos al nuevo?"*  

### **Evita Redundancias**:  
- Resume o parafrasea conceptos previamente explicados de forma breve.  

---

## **Idiomas**  
- Responde en el idioma en que se formula la pregunta.  
- Si se mezcla español e inglés, responde en el idioma predominante y ofrece traducciones si es útil.  

---

## **Transparencia y Límites**  
- Si no puedes proporcionar información específica:  
  - **Respuesta sugerida**:  
    *"No tengo información suficiente sobre este tema en mis recursos actuales. Por favor, consulta otras referencias especializadas."*  

---

## **Características Principales**  
1. **Respuestas Estructuradas Implícitamente**:  
   - Responde de manera fluida, organizando las ideas sin necesidad de secciones explícitas.  
2. **Priorización en Respuestas Largas**:  
   - Enfócate en conceptos clave y resume detalles secundarios.  
3. **Adaptabilidad a Preguntas Complejas**:  
   - Divide preguntas multifacéticas en respuestas claras y conectadas.  

---

## **Evaluación de Respuestas**  
Las respuestas deben ser:  
- **Relevantes**: Directamente relacionadas con la pregunta planteada.  
- **Claras**: Presentadas de manera lógica y accesible.  
- **Precisas**: Fundamentadas en las ideas de Mises y sus aplicaciones.  
- **Comprensibles**: Usando un lenguaje claro y enriquecedor.  

---

"""
)

def create_prompt_template3():
    return ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT3),
            MessagesPlaceholder(variable_name="history3"),
            ("human", "{question}")
        ]
    )

# Amazon Bedrock - KnowledgeBase Retriever 
retriever3 = AmazonKnowledgeBasesRetriever(
    knowledge_base_id="4L0WE8NOOH", # Set your Knowledge base ID
    retrieval_config={"vectorSearchConfiguration": {"numberOfResults": 20}},
)

model3 = ChatBedrock(
    client=bedrock_runtime,
    model_id=model_id,
    model_kwargs=model_kwargs,
)

prompt3 = create_prompt_template3()


chain3 = (
    RunnableParallel({
        "context": itemgetter("question") | retriever3,
        "question": itemgetter("question"),
        "history3": itemgetter("history3"),
    })
    .assign(response = prompt3 | model3 | StrOutputParser())
    .pick(["response", "context"])
)

# Streamlit Chat Message History
history3 = StreamlitChatMessageHistory(key="chat_messages3")

# Chain with History
chain_with_history3 = RunnableWithMessageHistory(
    chain3,
    lambda session_id: history3,
    input_messages_key="question",
    history_messages_key="history3",
    output_messages_key="response",
)






# ------------------------------------------------------
# Pydantic data model and helper function for Citations

class Citation(BaseModel):
    page_content: str
    metadata: Dict

def extract_citations(response: List[Dict]) -> List[Citation]:
    return [Citation(page_content=doc.page_content, metadata=doc.metadata) for doc in response]

# ------------------------------------------------------
# S3 Presigned URL, esto permite realizar descargar del documento

def create_presigned_url(bucket_name: str, object_name: str, expiration: int = 300) -> str:
    """Generate a presigned URL to share an S3 object"""
    s3_client = boto3.client('s3')
    try:
        response = s3_client.generate_presigned_url('get_object',
                                                    Params={'Bucket': bucket_name,
                                                            'Key': object_name},
                                                    ExpiresIn=expiration)
    except NoCredentialsError:
        st3.error("AWS credentials not available")
        return ""
    return response

def parse_s3_uri(uri: str) -> tuple:
    """Parse S3 URI to extract bucket and key"""
    parts = uri.replace("s3://", "").split("/")
    bucket = parts[0]
    key = "/".join(parts[1:])
    return bucket, key

# ------------------------------------------------------
# Streamlit


# Page title
st3.set_page_config(page_title='Chatbot CHH')
st3.subheader('Ludwig von Mises 🔗', divider='rainbow')

def display_history1(history):
    for message in history:
        content = message.content
        #safe_content = html.escape(message.content)  # Escapar HTML
        #html_content = markdown(message.content)  
        if message.__class__.__name__ == 'HumanMessage':  # Mensajes del usuario
            st3.markdown(
                f"""
                <div style="padding: 10px; margin-bottom: 10px; background-color: #0e1117; border-radius: 8px; color: #F3F4F6; font-size: 0.9em;">
                    <strong>Usuario:</strong><br>
                    {content}
                </div>
                """, unsafe_allow_html=True)
        elif message.__class__.__name__ == 'AIMessage':  # Respuestas del chatbot
            st3.markdown(
                f"""
                <div style="padding: 10px; margin-bottom: 10px; background-color: #0e1117; border-left: 5px solid #FF9F1C; border-radius: 8px; color: #E5E7EB; font-size: 0.9em;">
                    <strong>Chatbot (Ludwig von Mises):</strong><br>
                    {content}
                </div>
                """, unsafe_allow_html=True)

# Clear Chat History function
def clear_chat_history():
    history3.clear()
    st3.session_state.messages3 = [{"role": "assistant", "content": "Pregúntame sobre economía"}]

with st3.sidebar:
    streaming_on = True
    # st1.button('Limpiar chat', on_click=clear_chat_history)
    with st3.expander("Ver historial de conversación", expanded=False):  # collapsed por defecto
        display_history1(history3.messages) 

    st3.divider()

# Initialize session state for messages if not already present
if "messages3" not in st3.session_state:
    st3.session_state.messages3 = [{"role": "assistant", "content": "Pregúntame sobre economía"}]

# Display chat messages
for message in st3.session_state.messages3:
    with st3.chat_message(message["role"]):
        st3.write(message["content"])

# Chat Input - User Prompt 
if prompt := st3.chat_input():
    st3.session_state.messages3.append({"role": "user", "content": prompt})
    with st3.chat_message("user"):
        st3.write(prompt)

    config3 = {"configurable": {"session_id": "any"}}
    
    if streaming_on:
        # Chain - Stream
        with st3.chat_message("assistant"):
            placeholder3 = st3.empty()
            full_response3 = ''
            for chunk in chain_with_history3.stream(
                {"question" : prompt, "history3" : history3},
                config3
            ):
                if 'response' in chunk:
                    full_response3 += chunk['response']
                    placeholder3.markdown(full_response3)
                else:
                    full_context3 = chunk['context']
            placeholder3.markdown(full_response3)
            # Citations with S3 pre-signed URL
            citations3 = extract_citations(full_context3)
            with st3.expander("Mostrar fuentes >"):
                for citation in citations3:
                    st3.write("**Contenido:** ", citation.page_content)
                    s3_uri = citation.metadata['location']['s3Location']['uri']
                    
                    bucket, key = parse_s3_uri(s3_uri)
                    presigned_url = create_presigned_url(bucket, key)
                   ## if presigned_url:
                   ##         st3.markdown(f"**Fuente:** [{s3_uri}]({presigned_url})")
                   ## else:
                   ##         st3.write(f"**Fuente**: {s3_uri} (Presigned URL generation failed)")
                    st3.write(f"**Fuente**: *{key}* ")
                 
                    st3.write("**Score**:", citation.metadata['score'])
                    st3.write("--------------")

            # session_state append
            st3.session_state.messages3.append({"role": "assistant", "content": full_response3})
 