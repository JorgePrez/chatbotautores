

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

from langchain_community.chat_message_histories import DynamoDBChatMessageHistory
import uuid

from langchain.schema import HumanMessage, AIMessage

import streamlit_authenticator as stauth

# ------------------------------------------------------
# Log level

logging.getLogger().setLevel(logging.ERROR) # reduce log level

# ------------------------------------------------------

import streamlit as st



def authenticated_menu():
    # Show a navigation menu for authenticated users

    #st.sidebar.page_link("Todos los autores2.py", label= "Todos los autores")
    st.sidebar.page_link("app_autores2.py", label="Todos los autores")
    st.sidebar.page_link("pages/Hayek.py", label="Friedrich A. Hayek")
    st.sidebar.page_link("pages/Hazlitt.py", label="Henry Hazlitt")
    st.sidebar.page_link("pages/Mises.py", label="Ludwig von Mises")
    #st.divider()


def unauthenticated_menu():
    # Show a navigation menu for unauthenticated users
    st.sidebar.page_link("Todos los autores2.py", label="Log in")



def main():

    
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



    ############################################################

    dynamodb = boto3.resource("dynamodb", region_name="us-east-1")  # region
    table_name = "CHHSessionTable"  # Nombre de tu tabla DynamoDB

    # Clase para manejar el historial con formato específico
    class CustomDynamoDBChatMessageHistory:
        def __init__(self, table_name, session_id):
            self.table = dynamodb.Table(table_name)
            self.session_id = session_id

        def get_history(self):
            """Obtiene el historial completo desde DynamoDB."""
            response = self.table.get_item(Key={"SessionId": self.session_id})
            return response.get("Item", {"SessionId": self.session_id, "History": []})

        def update_history(self, new_message):
            """Actualiza el historial con un nuevo mensaje."""
            current_history = self.get_history()
            current_history["History"].append(new_message)

            # Guarda el historial actualizado en DynamoDB
            self.table.put_item(Item=current_history)


    # Función para crear el formato de mensaje
    def format_message(content, message_type="human", citations=None):
        """Crea un mensaje formateado con la estructura deseada."""
        data = {
            "additional_kwargs": {},
            "content": content,
            "example": False,
            "id": str(uuid.uuid4()),  # Genera un ID único para cada mensaje
            "name": None,
            "response_metadata": {},
            "type": message_type,
        }

        # Campos específicos para mensajes del asistente (AI)
        if message_type == "ai":
            data.update({
                "invalid_tool_calls": [],
                "tool_calls": [],
                "usage_metadata": None,
            })

        # Añadir citas si existen
        if citations:
            data["citations"] = citations

        return {"data": data, "type": message_type}



    ############################################################




    # Streamlit Chat Message History
    #history = StreamlitChatMessageHistory(key="chat_messages")



    # Chain with History


    #chain_with_history_dynamo = RunnableWithMessageHistory(
    #    chain,
    #    lambda session_id: DynamoDBChatMessageHistory(
    #        table_name=table_name, session_id=session_id
    #    ),
    #    input_messages_key="question",
    #    history_messages_key="history",
    #    output_messages_key="response",
    #)


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


    # Page title
    #st.set_page_config(page_title='Chatbot CHH')

    st.subheader('Todos los autores 🔗', divider='rainbow')
    streaming_on = True



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
                

    # Historial del chat

    table_name = "CHHSessionTable"

    history = StreamlitChatMessageHistory(key="chat_messages")

    # Chain with History, hay una cadena local, esta sirve para enviar al llm, ya que no guarda referencias
    chain_with_history = RunnableWithMessageHistory(
        chain,
        lambda session_id: history,
        input_messages_key="question",
        history_messages_key="history",
        output_messages_key="response",
    )


    # Crear instancia del historial
    #base_session_id = "placeholdercitations@gmail.com"  # Ejemplo de SessionId único
    base_session_id = st.session_state.username #generar_id_aleatorio()

    extra_identifier = "all_autores"
    # Concatenar el identificador adicional
    session_id = f"{base_session_id}-{extra_identifier}"
    chat_history = CustomDynamoDBChatMessageHistory(table_name=table_name, session_id=session_id)

    # Mostrar el historial en la barra lateral
    with st.sidebar:
        st.divider()
        st.title('Todos los autores 🔗')

        streaming_on = True
    # st.title('Historial de Conversación')
    # stored_history = chat_history.get_history()["History"]
    # for message in stored_history:
    #     role = "Usuario" if message["type"] == "human" else "Asistente"
    #     st.write(f"**{role}:** {message['data']['content']}")


        #########################################################################################


        # Llenando el history local, (esto es lo que se envia al LLM)
        history.clear()
        chat_history_data = chat_history.get_history()

        # Copiar mensajes al historial local (sin referencias)
        for message in chat_history_data.get("History", []):
            
                # Crear el objeto de mensaje adecuado
            if message["data"]["type"] == "human":
                msg_obj = HumanMessage(content=message["data"]["content"])
            else:
                msg_obj = AIMessage(content=message["data"]["content"])
                
            # Agregar al historial local
            #history.add_message(formatted_message["role"], formatted_message["content"])
            # Agregar el mensaje al historial local
            history.add_message(msg_obj)

        ########################################################################################
        with st.expander("Ver historial de conversación", expanded=False):  # collapsed por defecto
            display_history1(history.messages) 

        st.divider()


      
    ##########################################################################################
        # Llenando el session_state local
        if "messages" not in st.session_state:

            st.session_state.messages = []
        
            # Cargar los mensajes guardados de dynamo DB
            #stored_messages= chat_history.get_history()["History"] ##history.messages
            stored_messages = chat_history.get_history().get("History", [])  # Proveer una lista vacía si no hay historial

            if stored_messages:
 
                # Llenar el estado de la sesion con los mensajes obtenidos, importante que se utilizan el rol user / assistant

                for msg in stored_messages:
                # Determinar el rol
                    role = "user" if msg["data"]["type"] == "human" else "assistant"
            
                    # Crear el mensaje base con citations como campo separado
                    message = {
                        "role": role,
                        "content": msg["data"]["content"],
                        "citations": msg["data"].get("citations", [])  # Agregar citations si existen, de lo contrario una lista vacía
                    }
                        # Agregar al estado
                    st.session_state.messages.append(message)
            else :
                
                # Si no hay historial, mostrar mensaje inicial del asistente
                st.session_state.messages.append({"role": "assistant", "content": "Pregúntame sobre economía"})

            




    # Mostrar las preguntas anteriores con referencias en el chat
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            # Mostrar el contenido del mensaje
            st.write(message["content"])
            
            # Verificar si hay referencias y agregar un expander si existen
            if message.get("citations"):
                with st.expander("Mostrar referencias >"):
                    for citation in message["citations"]:
                        # Mostrar cada referencia con su contenido y fuente, este formato también puede ser utilizado
                    # st.write(f"- {citation['page_content']} (Fuente: {citation['metadata']['source']})")
                        # Mostrar cada referencia con su contenido y fuente
                        st.write(f" **Contenido:** {citation['page_content']} ")
                        st.write(f" **Fuente:** {citation['metadata']['source']}")
                       # st.write(f" **Score**: {citation['metadata']['score']}")
                        st.write("--------------")
                        score = (citation['metadata']['score'])
                        #st.write(f" **Score**: {(citation['metadata']['score'])}")
                    #  st.write("**Score**:", score)
                        #st.write("**Score**:", score)
        ######################################################################################



    # Chat Input - User Prompt
    if prompt := st.chat_input("Escribe tu mensaje aquí..."):
        # Mostrar el mensaje del usuario
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        #Esto sirve para el stream, pero no guarda en memoria, ya que para eso se utiliza el update history 
        # con la implementación propia de dynamoDB
        config = {"configurable": {"session_id": "any"}} #session_id

        if streaming_on:
            # Chain - Stream
            with st.chat_message("assistant"):
                placeholder = st.empty()
                full_response = ""
                
                # Iterar sobre los fragmentos del modelo
                for chunk in chain_with_history.stream(
                    {"question": prompt, "history": chat_history},
                    config
                ):
                    if "response" in chunk:
                        full_response += chunk["response"]
                        placeholder.markdown(full_response)
                    else:
                        full_context = chunk["context"]
                
                # Mostrar la respuesta completa
                placeholder.markdown(full_response)
                
                # Extraer citas y generar URLs pre-firmadas si es necesario
                citations = extract_citations(full_context)
                formatted_citations = []  # Lista para almacenar las citas en el formato deseado
                with st.expander("Mostrar referencias >"):
                    for citation in citations:
                        st.write("**Contenido:** ", citation.page_content)
                        source = ""
                        score =""
                        if "location" in citation.metadata and "s3Location" in citation.metadata["location"]:
                            s3_uri = citation.metadata["location"]["s3Location"]["uri"]
                            bucket, key = parse_s3_uri(s3_uri)
                            st.write(f"**Fuente**: *{key}* ")
                            source = key
                            score= citation.metadata['score']
                           # st.write("**Score**:", score)

                        else:
                            st.write("**Fuente:** No disponible")
                        st.write("--------------")

                                            # Agregar al formato de placeholder_citations
                        formatted_citations.append({
                                "page_content": citation.page_content,
                                "metadata": {
                                    "source": source,
                                    "score" : str(score)
                                }
                            })
                
                # placeholder_citations = [
                #{"page_content": "Contenido de ejemplo 1", "metadata": {"source": "Fuente 1"}},
                #{"page_content": "Contenido de ejemplo 2", "metadata": {"source": "Fuente 2"}}  ]
                
                # Guardar mensajes en DynamoDB con la estructura deseada
                # Crear el mensaje del usuario
                human_message = format_message(prompt, "human")
                chat_history.update_history(human_message)

                # Crear el mensaje del asistente(chatbot) con citas
                ai_message = format_message(full_response, "ai", formatted_citations)
                chat_history.update_history(ai_message)


            # st.session_state.messages.append(message)
            # Hacer append del primer mensaje

            
                #session_state con referencias
                st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
                "citations": formatted_citations  # Guardar referencias junto con la respuesta.
            })
                

def authenticator_login():


    
    import yaml
    from yaml.loader import SafeLoader
    with open('userschh.yaml') as file:
        config = yaml.load(file, Loader=SafeLoader)


    # Inicializar el estado del botón si no existe
    if "show_register_form" not in st.session_state:
        st.session_state["show_register_form"] = False



    ## st.set_page_config(page_title='Procesos UFM 🔗')
    st.set_page_config(page_title='Chatbot CHH')


    # Pre-hashing all plain text passwords once
    #stauth.Hasher.hash_passwords(config['credentials'])

    authenticator = stauth.Authenticate(
        config['credentials'],
        config['cookie']['name'],
        config['cookie']['key'],
        config['cookie']['expiry_days']
    )

    # Example Microsoft login widget
    #try:
    #    authenticator.experimental_guest_login('Login with Microsoft',
    #                                       provider='microsoft',
    #                                       oauth2=config['oauth2'])
    #                    # Guardar la nueva configuración
    #    with open('userschh.yaml', 'w') as file:
    #                yaml.dump(config, file, default_flow_style=False)
    #except Exception as e:
    #    st.error(e)

    
    # Example Google login widget
    try:
        authenticator.experimental_guest_login('Login with Google',
                                           provider='google',
                                           oauth2=config['oauth2'])
        
                        # Guardar la nueva configuración
        with open('userschh.yaml', 'w') as file:
                    yaml.dump(config, file, default_flow_style=False)

    except Exception as e:
        st.error(e)






if __name__ == "__main__":
    authenticator_login()
