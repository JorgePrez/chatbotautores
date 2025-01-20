

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

from markdown import markdown
import html

# ------------------------------------------------------
# Log level

logging.getLogger().setLevel(logging.ERROR) # reduce log level

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

prompt_old = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant, always answer in Spanish"
         "Answer the question based only on the following context:\n {context}"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)

SYSTEM_PROMPTALL = (
"""
### Base de conocimientos:  
{context}  

---

# Prompt del Sistema: Chatbot Especializado en Hazlitt, Mises y Hayek  

## **Identidad del Asistente**  
Eres un asistente virtual especializado en proporcionar explicaciones claras y detalladas sobre los principales conceptos y teorías de **Henry Hazlitt**, **Ludwig von Mises** y **Friedrich A. Hayek**. Tu propósito es facilitar el aprendizaje autónomo y la comprensión de sus contribuciones a la filosofía económica, con énfasis en la Escuela Austriaca de Economía. Respondes en español e inglés, adaptándote a las necesidades del usuario.  

Puedes responder desde la perspectiva de uno o más de estos autores, según sea relevante para la pregunta, conectando sus ideas cuando corresponda.  

## **Público Objetivo**  
### **Audiencia Primaria**:  
- **Estudiantes** (de 18 a 45 años) de la **Universidad Francisco Marroquín (UFM)** en Guatemala.  
- Carreras: economía, derecho, ciencias políticas, filosofía, administración de empresas y otras relacionadas.  
- Principal enfoque en estudiantes de pregrado interesados en la Escuela Austriaca y sus aplicaciones.  

### **Audiencia Secundaria**:  
- Profesores y académicos que deseen integrar las ideas de Hazlitt, Mises y Hayek en sus debates sobre política económica y filosofía política.  

### **Audiencia Terciaria**:  
- Economistas, empresarios y entusiastas de la economía en **Latinoamérica, España**, y otras regiones interesados en los mercados libres, la crítica al socialismo, y las teorías del orden espontáneo, el cálculo económico y el análisis de políticas públicas.  

---

## **Metodología para Respuestas**  
Las respuestas deben seguir una estructura lógica basada en la metodología **5W 1H** (qué, quién, cuándo, dónde, por qué, cómo). Deben integrar las ideas de uno, dos o los tres autores según la relevancia para la pregunta planteada.  

- **Introduce el tema o concepto de manera clara y directa.**  
- Amplía con definiciones, ejemplos históricos y aplicaciones contemporáneas, vinculando las perspectivas de Hazlitt, Mises y Hayek donde sea pertinente.  
- Finaliza con reflexiones o conexiones relevantes al tema.  

---

## **Estructura Implícita de Respuesta**  
1. **Contexto inicial**: Presentar el tema con énfasis en su relevancia y las contribuciones de los autores.  
2. **Desarrollo de ideas**: Explorar conceptos clave, ejemplos prácticos y aplicaciones modernas desde una o más perspectivas.  
3. **Cierre reflexivo**: Resumir la idea principal y conectar con implicaciones actuales o debates relevantes.  

---

## **Tono y Estilo**  
- **Profesional y académico**, con un enfoque claro y motivador.  
- Lenguaje preciso y accesible, libre de tecnicismos innecesarios.  
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
- **Precisas**: Fundamentadas en las ideas de Hazlitt, Mises y Hayek según sea relevante.  
- **Comprensibles**: Usando un lenguaje claro y enriquecedor.  

---

"""
)


def create_prompt_template_all():
    return ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPTALL),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}")
        ]
    )

# Amazon Bedrock - KnowledgeBase Retriever 
retriever = AmazonKnowledgeBasesRetriever(
    knowledge_base_id="WGUUTHDVPH", #  Knowledge base ID
    retrieval_config={"vectorSearchConfiguration": {"numberOfResults": 20}},
)

model = ChatBedrock(
    client=bedrock_runtime,
    model_id=model_id,
    model_kwargs=model_kwargs,
)

prompt_all = create_prompt_template_all()

chain = (
    RunnableParallel({
        "context": itemgetter("question") | retriever,
        "question": itemgetter("question"),
        "history": itemgetter("history"),
    })
    .assign(response = prompt_all | model | StrOutputParser())
    .pick(["response", "context"])
)

# Streamlit Chat Message History
history = StreamlitChatMessageHistory(key="chat_messages")

# Chain with History
chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: history,
    input_messages_key="question",
    history_messages_key="history",
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
        st.error("AWS credentials not available")
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

import streamlit as st

# Page title
st.set_page_config(page_title='Chatbot CHH')

st.subheader('Todos los autores 🔗', divider='rainbow')


# Función para formatear el historial

def display_history1(history):
    for message in history:
        content = message.content
        #safe_content = html.escape(message.content)  # Escapar HTML
        #html_content = markdown(message.content)  
        if message.__class__.__name__ == 'HumanMessage':  # Mensajes del usuario
            st.markdown(
                f"""
                <div style="padding: 10px; margin-bottom: 10px; background-color: #0e1117; border-radius: 8px; color: #F3F4F6; font-size: 0.9em;">
                    <strong>Usuario:</strong><br>
                    {content}
                </div>
                """, unsafe_allow_html=True)
        elif message.__class__.__name__ == 'AIMessage':  # Respuestas del chatbot
            st.markdown(
                f"""
                <div style="padding: 10px; margin-bottom: 10px; background-color: #0e1117; border-left: 5px solid #FF9F1C; border-radius: 8px; color: #E5E7EB; font-size: 0.9em;">
                    <strong>Chatbot (Todos los autores):</strong><br>
                    {content}
                </div>
                """, unsafe_allow_html=True)
            

## Ejemplo de lo que esta en history.messages
history_placeholders = [
    type('HumanMessage', (object,), {"content": "¿Cuáles son las similitudes clave entre Hayek y Mises?"})(),
    type('AIMessage', (object,), {"content": "Ambos fueron defensores del libre mercado y críticos del intervencionismo estatal..."})()
]


def show_modal(citations):
    st.markdown(
        """
        <style>
        .modal {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0, 0, 0, 0.8);
            display: flex;
            justify-content: center;
            align-items: center;
            z-index: 1000;
        }
        .modal-content {
            background-color: #1e293b;
            padding: 20px;
            border-radius: 8px;
            max-width: 600px;
            width: 90%;
            color: #f3f4f6;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
            overflow-y: auto;
            max-height: 80%;
        }
        .close-btn {
            color: white;
            float: right;
            font-size: 20px;
            font-weight: bold;
            cursor: pointer;
        }
        </style>
        """, unsafe_allow_html=True
    )
    st.markdown(
        f"""
        <div class="modal">
            <div class="modal-content">
                <span class="close-btn" onclick="document.querySelector('.modal').style.display='none'">&times;</span>
                <h3>Referencias relacionadas</h3>
                <ul>
                    {"".join([f"<li>{citation['page_content']}</li>" for citation in citations])}
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True
    ) 

# Clear Chat History function
def clear_chat_history():
    history.clear()
    st.session_state.messages = [{"role": "assistant", "content": "Pregúntame sobre economía"}]

with st.sidebar:
    st.title('Todos los autores 🔗')

    streaming_on = True

    with st.expander("Ver historial de conversación", expanded=False):  # collapsed por defecto
        display_history1(history.messages) 

    st.divider()

   

# Initialize session state for messages if not already present
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Pregúntame sobre economía"}]

# Display chat messages

#for message in st.session_state.messages:
#    with st.chat_message(message["role"]):
#        st.write(message["content"])


# Mostrar historial de chat con referencias
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        
        # Mostrar referencias si existen
        if "citations" in message and message["citations"]:
            with st.expander("Mostrar referencias >"):
                for citation in message["citations"]:
                    st.write(f"**Contenido:** {citation.page_content}")
                    s3_uri = citation.metadata['location']['s3Location']['uri']
                    bucket, key = parse_s3_uri(s3_uri)
                    st.write(f"**Fuente:** *{key}*")
                    st.write("--------------")

# Chat Input - User Prompt 
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    config = {"configurable": {"session_id": "any"}}
    
    if streaming_on:
        # Chain - Stream
        with st.chat_message("assistant"):
            placeholder = st.empty()
            full_response = ''
            for chunk in chain_with_history.stream(
                {"question" : prompt, "history" : history},
                config
            ):
                if 'response' in chunk:
                    full_response += chunk['response']
                    placeholder.markdown(full_response)
                else:
                    full_context = chunk['context']
            placeholder.markdown(full_response)
            # Citations with S3 pre-signed URL
            citations = extract_citations(full_context)
            with st.expander("Mostrar referencias >"):
                for citation in citations:
                    st.write("**Contenido:** ", citation.page_content)
                    s3_uri = citation.metadata['location']['s3Location']['uri']
                    bucket, key = parse_s3_uri(s3_uri)

                      #  presigned_url = create_presigned_url(bucket, key)
                     #   if presigned_url:
                     #       st.markdown(f"**Fuente:** [{s3_uri}]({presigned_url})")
                     #   else:
                      #  st.write(f"**Fuente**: {s3_uri} (Presigned URL generation failed)")
                    st.write(f"**Fuente**: *{key}* ")
                    st.write("--------------")

            # session_state append
            #st.session_state.messages.append({"role": "assistant", "content": full_response})


            #session_state con referencias
            st.session_state.messages.append({
            "role": "assistant",
            "content": full_response,
            "citations": citations  # Guardar referencias junto con la respuesta.
        })
            
        #print(st.session_state)

  

