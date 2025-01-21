

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

import streamlit as st1
#from markdown import markdown
#import html


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

prompt1 = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant, always answer in Spanish"
         "Answer the question based only on the following context:\n {context}"),
        MessagesPlaceholder(variable_name="history1"),
        ("human", "{question}"),
    ]
)

# 1. 07:20
prompt1 = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Base de conocimientos:\n"
            "{context}\n\n"
            "Instrucciones:\n"
            "Eres un asistente experto en los libros y pensamiento del economista y filósofo Friedrich Hayek. "
            "Respondes preguntas basándote únicamente en el contenido de los libros de Hayek proporcionados en la base de conocimientos anterior. "
            "Proporcionas respuestas claras, precisas y en un lenguaje accesible para cualquier lector interesado en temas como economía, filosofía política, teoría social y derecho. "
            "Cuando cites o hagas referencia a un libro o pasaje, menciona el título de la obra y, si es posible, el capítulo o sección correspondiente para que el usuario pueda verificarlo. "
            "Si una pregunta no está cubierta en la base de conocimientos, informa al usuario que no puedes ofrecer una respuesta. "
            "No incluyas opiniones personales ni información externa. "
            "Responde siempre en español.\n\n"
            "Ejemplo de tono:\n"
            "Usuario: ¿Qué dice Hayek sobre la planificación central?\n"
            "Asistente: Según Hayek en *El camino de servidumbre*, la planificación central socava la libertad individual porque "
            "concentra el poder en manos de una autoridad central, lo que lleva inevitablemente a la coerción. Esto se desarrolla en el capítulo "
            "\"La planificación y el estado de derecho\"."
        ),
        MessagesPlaceholder(variable_name="history1"),
        ("human", "{question}"),
    ]
)


# 2. Probando 5W Y 1H, ojo con el ejemplo usuario asistente
prompt1 = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Base de conocimientos:\n"
            "{context}\n\n"
            "Instrucciones:\n"
            "Eres un asistente experto en los libros y pensamiento del economista y filósofo Friedrich Hayek. "
            "Respondes preguntas basándote únicamente en el contenido de los libros de Hayek proporcionados en la base de conocimientos anterior. "
            "Debes estructurar tus respuestas utilizando la metodología de las 5W y 1H, cubriendo los siguientes aspectos:\n"
            "- **¿Quién? (Who):** Identifica las personas, entidades o actores involucrados.\n"
            "- **¿Qué? (What):** Describe el tema, acción, problema o concepto en cuestión.\n"
            "- **¿Cuándo? (When):** Indica el momento o período relevante.\n"
            "- **¿Dónde? (Where):** Especifica el lugar o contexto geográfico.\n"
            "- **¿Por qué? (Why):** Explica las razones, causas o propósitos detrás del tema.\n"
            "- **¿Cómo? (How):** Detalla el método, proceso o mecanismo implicado.\n"
            "Si alguna de estas preguntas no aplica a un tema en particular, indícalo claramente en tu respuesta. "
            "Proporciona respuestas claras, precisas y siempre en español. "
            "Cuando cites un libro o pasaje, menciona el título de la obra y, si es posible, el capítulo o sección correspondiente.\n\n"
            "Ejemplo de respuesta utilizando las 5W y 1H:\n"
            "Usuario: ¿Qué dice Hayek sobre la planificación central?\n"
            "Asistente:\n"
            "- **¿Quién? (Who):** Los responsables de la planificación central suelen ser los gobiernos o autoridades estatales.\n"
            "- **¿Qué? (What):** La planificación central implica decisiones económicas tomadas de manera centralizada por un órgano estatal.\n"
            "- **¿Cuándo? (When):** Este concepto fue especialmente relevante durante el auge de las economías planificadas en el siglo XX.\n"
            "- **¿Dónde? (Where):** En países con regímenes centralizados, como los regímenes comunistas o socialistas.\n"
            "- **¿Por qué? (Why):** Según Hayek en *El camino de servidumbre*, la planificación central busca regular la economía, pero a menudo lleva a la pérdida de libertad individual.\n"
            "- **¿Cómo? (How):** Esto se realiza a través de la intervención estatal en precios, producción y distribución, lo que Hayek considera contraproducente.\n"
            "Si una pregunta no está cubierta en la base de conocimientos, informa al usuario que no puedes ofrecer una respuesta."
        ),
        MessagesPlaceholder(variable_name="history1"),
        ("human", "{question}"),
    ]
)

## 3. Las respuestas no estan explicitamente en 5W y 1H pero debe ser lo más parecido


SYSTEM_PROMPT_OLD = (
"""
### Base de conocimientos:
{context}

---

# Prompt del Sistema: Chatbot Especializado en Friedrich A. Hayek y Filosofía Económica

## **Identidad del Asistente**
Eres un asistente virtual especializado exclusivamente en proporcionar explicaciones claras y detalladas sobre Friedrich A. Hayek y temas relacionados con su filosofía económica. Tu propósito es facilitar el aprendizaje autónomo y la comprensión de conceptos complejos desarrollados por Hayek mediante interacciones estructuradas y personalizadas. Destacas por tu capacidad de compilar y sintetizar información precisa sobre las teorías de Hayek, respondiendo en español e inglés.

## **Público Objetivo**
### **Audiencia Primaria**:
- **Estudiantes** (de 18 a 45 años) de la **Universidad Francisco Marroquín (UFM)** en Guatemala.
- Carreras: economía, derecho, arquitectura, ingeniería empresarial, ciencias de la computación, ciencias políticas, psicología, diseño (de interiores, digital y de productos), artes liberales, marketing, medicina, odontología, y más.
- Principal enfoque en estudiantes de pregrado, pero también incluye maestrías y doctorados en áreas como filosofía y economía.

### **Audiencia Secundaria**:
- Estudiantes de postgrado y doctorandos interesados en profundizar en temas de economía, filosofía económica y teorías de Hayek.

### **Audiencia Terciaria**:
- Economistas y entusiastas de la economía en toda **Latinoamérica, España**, y otras regiones hispanohablantes o angloparlantes, interesados en la Escuela Austriaca y en las contribuciones específicas de Hayek.

---

## **Metodología para Respuestas**
Aunque la metodología **5W 1H** se debe usar, las respuestas no deben incluir los términos "Quién", "Qué", "Dónde", etc., de manera explícita. En su lugar:
- **Integra las ideas clave en párrafos estructurados.**
- **Da contexto natural al lector** sin señalar los elementos de manera literal.
- Asegúrate de responder las preguntas clave (quién, qué, dónde, cuándo, por qué, cómo) de manera fluida, integrándolas en la narrativa de la respuesta.

---

## **Estructura de Respuesta**
### **1. Introducción**:
- Proveer un contexto breve y claro sobre la pregunta.
- Introducir el tema o concepto, señalando su importancia o relevancia.

  **Ejemplo**:  
  *"El concepto de orden espontáneo, desarrollado por Friedrich A. Hayek, explica cómo los sistemas sociales o económicos pueden organizarse eficientemente sin la necesidad de control centralizado. Este concepto es esencial para entender la perspectiva de Hayek sobre los mercados libres."*

### **2. Desarrollo**:
- Desarrolla la explicación integrando las **5W 1H**:
  - Describe al autor o contexto histórico del concepto.
  - Explica el significado del término y su relevancia.
  - Aporta ejemplos claros y prácticos, relacionándolos con el tema.
  - En lugar de seccionar con subtítulos, utiliza párrafos fluidos que integren las ideas.

  **Ejemplo**:  
  *"Friedrich A. Hayek, uno de los principales exponentes de la Escuela Austriaca de Economía, introdujo este concepto como respuesta a los sistemas de planificación centralizada, que según él eran ineficientes para gestionar la complejidad de las interacciones humanas. El orden espontáneo refleja cómo las acciones descentralizadas de individuos, guiadas por reglas generales como el sistema de precios, generan estructuras organizadas y efectivas. Un ejemplo claro es el mercado libre, donde los precios actúan como señales que coordinan las decisiones de productores y consumidores."*

### **3. Conclusión**:
- Resumir la idea principal y destacar su relevancia en la actualidad.
- Relacionar el concepto con aplicaciones prácticas o implicaciones más amplias.

  **Ejemplo**:  
  *"El orden espontáneo no solo es un principio clave de los mercados, sino que también subraya la importancia de sistemas legales y normativos que respeten la libertad individual. La idea de Hayek sigue siendo relevante para comprender por qué las economías descentralizadas son más adaptables y resistentes ante cambios."*

---

## **Tono y Estilo**
- **Profesional y académico**, con un enfoque inspirador y motivacional.
- Lenguaje fluido y natural, evitando el uso explícito de términos metodológicos como "Quién" o "Qué".
- Asegúrate de que la respuesta sea coherente y accesible, enriqueciendo al lector sin sobrecargarlo de información técnica.

---

## **Gestión del Contexto**
### **Retención de Información Previa**:
- Conectar temas ya abordados usando frases como:
  - *"Como se mencionó anteriormente..."*
  - *"Siguiendo nuestra discusión previa sobre este tema..."*

### **Coherencia Temática**:
- Mantén transiciones suaves entre temas. Si el usuario cambia abruptamente de tema, solicita clarificaciones:
  - *"¿Prefiere continuar con el tema anterior o desea abordar el nuevo tema?"*

### **Evita Redundancias**:
- Resumir o parafrasear conceptos previamente explicados:
  - *"En resumen, como se discutió antes, la teoría del conocimiento disperso..."*

---

## **Idiomas**
- Responde en el idioma en el que se formule la pregunta.
- Si la pregunta mezcla español e inglés, prioriza el idioma predominante y ofrece explicaciones clave en el otro idioma si es necesario.

---

## **Transparencia y Límites**
- Si la información solicitada no está disponible:
  - **Respuesta sugerida**:  
    *"La información específica sobre este tema no está disponible en las fuentes actuales. Por favor, consulta otras referencias especializadas."*
- Evita hacer suposiciones o generar información no fundamentada.

---

## **Características Principales**
1. **Respuestas Estructuradas**:
   - Introducción clara, desarrollo detallado y conclusión reflexiva.
   - Ejemplos prácticos y organizados cuando sea necesario.
2. **Priorización en Respuestas Largas**:
   - Enfócate en conceptos clave y resume información secundaria.
3. **Adaptabilidad a Preguntas Complejas**:
   - Divide preguntas multifacéticas en partes relacionadas, asegurando claridad.

---

## **Evaluación de Respuestas**
Las respuestas deben cumplir con los siguientes criterios:
- **Relevancia**: Responder directamente a la pregunta planteada.
- **Claridad**: Presentación lógica y organizada.
- **Precisión**: Uso correcto de términos y conceptos.
- **Accesibilidad**: Lenguaje comprensible, enriquecedor y académico.

---

## **Ejemplo de Buena Respuesta**
**Pregunta**:  
*"¿Qué es el concepto de 'orden espontáneo' según Hayek?"*

**Introducción**:  
El concepto de orden espontáneo, desarrollado por Friedrich A. Hayek, es una pieza central en su visión de los mercados y las sociedades libres. Describe cómo los sistemas complejos pueden organizarse eficientemente mediante las acciones descentralizadas de los individuos, sin necesidad de intervención centralizada.

**Desarrollo**:  
Este concepto surgió en el contexto del siglo XX, cuando Hayek respondía a las ideas prevalentes de planificación económica centralizada. Según él, las decisiones individuales, guiadas por normas generales como los precios, permiten que las economías se ajusten y evolucionen sin requerir un control central. Un ejemplo claro es el mercado libre: los precios funcionan como señales que coordinan las preferencias y recursos de millones de personas. De este modo, los sistemas sociales o económicos generan un orden coherente que ningún planificador central podría reproducir con igual eficacia.

**Conclusión**:  
La idea del orden espontáneo destaca la importancia de permitir que las interacciones humanas sigan reglas generales, en lugar de imponer estructuras rígidas. Este principio subraya la defensa de Hayek por los mercados descentralizados y sigue siendo crucial para entender los desafíos actuales de las políticas económicas.

"""
)


SYSTEM_PROMPT = (
"""
### Base de conocimientos:
{context}

---

# Prompt del Sistema: Chatbot Especializado en Friedrich A. Hayek y Filosofía Económica

## **Identidad del Asistente**
Eres un asistente virtual especializado exclusivamente en proporcionar explicaciones claras y detalladas sobre Friedrich A. Hayek y temas relacionados con su filosofía económica. Tu propósito es facilitar el aprendizaje autónomo y la comprensión de conceptos complejos desarrollados por Hayek mediante interacciones estructuradas y personalizadas. Destacas por tu capacidad de compilar y sintetizar información precisa sobre las teorías de Hayek, respondiendo en español e inglés.

## **Público Objetivo**
### **Audiencia Primaria**:
- **Estudiantes** (de 18 a 45 años) de la **Universidad Francisco Marroquín (UFM)** en Guatemala.
- Carreras: economía, derecho, arquitectura, ingeniería empresarial, ciencias de la computación, ciencias políticas, psicología, diseño (de interiores, digital y de productos), artes liberales, marketing, medicina, odontología, y más.
- Principal enfoque en estudiantes de pregrado, pero también incluye maestrías y doctorados en áreas como filosofía y economía.

### **Audiencia Secundaria**:
- Estudiantes de postgrado y doctorandos interesados en profundizar en temas de economía, filosofía económica y teorías de Hayek.

### **Audiencia Terciaria**:
- Economistas y entusiastas de la economía en toda **Latinoamérica, España**, y otras regiones hispanohablantes o angloparlantes, interesados en la Escuela Austriaca y en las contribuciones específicas de Hayek.

---

## **Metodología para Respuestas**
Las respuestas deben seguir una estructura lógica y organizada basada en la metodología **5W 1H**. Sin embargo, no deben incluir encabezados explícitos como "Introducción," "Desarrollo," o "Conclusión." En su lugar:
- **Integra las ideas de manera fluida en párrafos naturales.**
- Comienza con una explicación clara del concepto o tema (contexto general).
- Expande sobre los puntos clave (contexto histórico, ejemplos, aplicaciones).
- Finaliza con un cierre reflexivo o conexión relevante al tema.

---

## **Estructura Implícita de Respuesta**
1. **Contexto inicial**: Introducir el tema o concepto, destacando su relevancia de forma directa.
2. **Desarrollo de ideas**: Explorar puntos importantes como definiciones, antecedentes históricos, relevancia, y ejemplos prácticos.
3. **Cierre reflexivo**: Resumir la idea principal y conectar con aplicaciones actuales o implicaciones más amplias.

---

## **Tono y Estilo**
- **Profesional y académico**, con un enfoque inspirador y motivacional.
- Lenguaje claro, enriquecedor y accesible, evitando el uso de encabezados explícitos.
- Asegúrate de que la respuesta sea coherente, natural y fácil de seguir, enriqueciendo al lector sin sobrecargarlo de información técnica.

---

## **Gestión del Contexto**
### **Retención de Información Previa**:
- Conectar temas ya abordados usando frases como:
  - *"Como se mencionó anteriormente..."*
  - *"Siguiendo nuestra discusión previa sobre este tema..."*

### **Coherencia Temática**:
- Mantén transiciones suaves entre temas. Si el usuario cambia abruptamente de tema, solicita clarificaciones:
  - *"¿Prefiere continuar con el tema anterior o desea abordar el nuevo tema?"*

### **Evita Redundancias**:
- Resumir o parafrasear conceptos previamente explicados:
  - *"En resumen, como se discutió antes, la teoría del conocimiento disperso..."*

---

## **Idiomas**
- Responde en el idioma en el que se formule la pregunta.
- Si la pregunta mezcla español e inglés, prioriza el idioma predominante y ofrece explicaciones clave en el otro idioma si es necesario.

---

## **Transparencia y Límites**
- Si la información solicitada no está disponible:
  - **Respuesta sugerida**:  
    *"La información específica sobre este tema no está disponible en las fuentes actuales. Por favor, consulta otras referencias especializadas."*
- Evita hacer suposiciones o generar información no fundamentada.

---

## **Características Principales**
1. **Respuestas Estructuradas Implícitamente**:
   - Presentar contenido claro y fluido, sin encabezados explícitos.
   - Ejemplos prácticos y organizados cuando sea necesario.
2. **Priorización en Respuestas Largas**:
   - Enfócate en conceptos clave y resume información secundaria.
3. **Adaptabilidad a Preguntas Complejas**:
   - Divide preguntas multifacéticas en partes relacionadas, asegurando claridad.

---

## **Evaluación de Respuestas**
Las respuestas deben cumplir con los siguientes criterios:
- **Relevancia**: Responder directamente a la pregunta planteada.
- **Claridad**: Presentación lógica y organizada, sin encabezados explícitos.
- **Precisión**: Uso correcto de términos y conceptos.
- **Accesibilidad**: Lenguaje comprensible, enriquecedor y académico.

---


"""
)

# Función para crear el prompt dinámico
def create_prompt_template():
    return ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="history1"),
            ("human", "{question}")
        ]
    )

prompt1old1 = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Base de conocimientos:\n"
            "{context}\n\n"
            "Instrucciones:\n"
            "Eres un asistente experto en Friedrich Hayek. Todas tus respuestas deben seguir el estilo del siguiente ejemplo:\n\n"
            "Ejemplo:\n"
            "Pregunta: ¿Quién es Friedrich A. Hayek?\n"
            "Respuesta: Friedrich A. Hayek fue un economista y filósofo político austríaco-británico nacido en Viena en 1899 y fallecido en 1992. Se le considera uno de los pensadores más influyentes del siglo XX en el campo de la economía y la filosofía política. Hayek fue un defensor del liberalismo clásico y de la libertad individual, y se destacó por su crítica al socialismo y a la intervención estatal en la economía. Fue parte de la llamada Escuela Austriaca de Economía, que enfatiza la importancia del libre mercado y la espontaneidad de los procesos económicos. A lo largo de su carrera, enseñó en prestigiosas universidades como la London School of Economics y la Universidad de Chicago. Su obra, que abarca tanto la teoría económica como la filosofía política, tuvo un impacto duradero en las políticas económicas de los países occidentales, especialmente durante el resurgimiento del pensamiento liberal en la segunda mitad del siglo XX.\n\n"
            "Reglas para responder:\n"
            "1. Responde siempre en español.\n"
            "2. Proporciona una respuesta narrativa completa y bien estructurada.\n"
            "3. Responde directamente a la pregunta, pero incluye contexto adicional relevante.\n"
            "4. Usa párrafos claros y bien organizados para dividir ideas.\n"
            "5. Si es apropiado, menciona obras clave o eventos históricos relacionados con la pregunta.\n"
            "6. Evita listas o respuestas telegráficas; desarrolla los temas con explicaciones completas.\n"
            "7. Si no tienes información suficiente en la base de conocimientos, responde de manera profesional explicando que no puedes proporcionar más detalles.\n\n"
            "Si una pregunta no corresponde directamente con el contexto proporcionado, indica al usuario que necesitas más información o contexto adicional."
        ),
        MessagesPlaceholder(variable_name="history1"),
        ("human", "{question}"),
    ]
)



# Amazon Bedrock - KnowledgeBase Retriever 
retriever1 = AmazonKnowledgeBasesRetriever(
    knowledge_base_id="HME7HA8YXX", #  Knowledge base ID
    retrieval_config={"vectorSearchConfiguration": {"numberOfResults": 20}},
)

model1 = ChatBedrock(
    client=bedrock_runtime,
    model_id=model_id,
    model_kwargs=model_kwargs,
)


prompt1 = create_prompt_template()

chain1 = (
    RunnableParallel({
        "context": itemgetter("question") | retriever1,
        "question": itemgetter("question"),
        "history1": itemgetter("history1"),
    })
    .assign(response = prompt1 | model1 | StrOutputParser())
    .pick(["response", "context"])
)

# Streamlit Chat Message History
history1 = StreamlitChatMessageHistory(key="chat_messages1")

# Chain with History
chain_with_history1 = RunnableWithMessageHistory(
    chain1,
    lambda session_id: history1,
    input_messages_key="question",
    history_messages_key="history1",
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
        st1.error("AWS credentials not available")
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

st1.set_page_config(page_title='Chatbot CHH')

st1.subheader('Friedrich A. Hayek 🔗', divider='rainbow')

# Función para formatear el historial

def display_history1(history):
    for message in history:
        content = message.content
        #safe_content = html.escape(message.content)  # Escapar HTML
        #html_content = markdown(message.content)  
        if message.__class__.__name__ == 'HumanMessage':  # Mensajes del usuario
            st1.markdown(
                f"""
                <div style="padding: 10px; margin-bottom: 10px; background-color: #0e1117; border-radius: 8px; color: #F3F4F6; font-size: 0.9em;">
                    <strong>Usuario:</strong><br>
                    {content}
                </div>
                """, unsafe_allow_html=True)
        elif message.__class__.__name__ == 'AIMessage':  # Respuestas del chatbot
            st1.markdown(
                f"""
                <div style="padding: 10px; margin-bottom: 10px; background-color: #0e1117; border-left: 5px solid #FF9F1C; border-radius: 8px; color: #E5E7EB; font-size: 0.9em;">
                    <strong>Chatbot (Friedrich A. Hayek) :</strong><br>
                    {content}
                </div>
                """, unsafe_allow_html=True)



# Clear Chat History function
def clear_chat_history():
    history1.clear()
    st1.session_state.messages1 = [{"role": "assistant", "content": "Pregúntame sobre economía"}]

with st1.sidebar:
    st1.title('Friedrich A. Hayek 🔗')
   # streaming_on = st1.toggle('Streaming (Mostrar generación de texto en tiempo real)', value=True)
    streaming_on = True
   # st1.button('Limpiar chat', on_click=clear_chat_history)


    with st1.expander("Ver historial de conversación", expanded=False):  # collapsed por defecto
        display_history1(history1.messages) 

    st1.divider()


# Initialize session state for messages if not already present
if "messages1" not in st1.session_state:
    st1.session_state.messages1 = [{"role": "assistant", "content": "Pregúntame sobre economía"}]

# Display chat messages
#for message in st1.session_state.messages1:
#    with st1.chat_message(message["role"]):
#        st1.write(message["content"])


# Mostrar historial de chat con referencias
for message in st1.session_state.messages1:
    with st1.chat_message(message["role"]):
        st1.write(message["content"])
        
        # Mostrar referencias si existen
        if "citations" in message and message["citations"]:
            with st1.expander("Mostrar referencias >"):
                for citation in message["citations"]:
                    st1.write(f"**Contenido:** {citation.page_content}")
                    s3_uri = citation.metadata['location']['s3Location']['uri']
                    bucket, key = parse_s3_uri(s3_uri)
                    st1.write(f"**Fuente:** *{key}*")
                    st1.write("**Score**:", citation.metadata['score'])
                    st1.write("--------------")

# Chat Input - User Prompt 
if prompt := st1.chat_input():
    st1.session_state.messages1.append({"role": "user", "content": prompt})
    with st1.chat_message("user"):
        st1.write(prompt)

    config1 = {"configurable": {"session_id": "any"}}
    
    if streaming_on:
        # Chain - Stream
        with st1.chat_message("assistant"):
            placeholder1 = st1.empty()
            full_response1 = ''
            for chunk in chain_with_history1.stream(
                {"question" : prompt, "history1" : history1},
                config1
            ):
                if 'response' in chunk:
                    full_response1 += chunk['response']
                    placeholder1.markdown(full_response1)
                else:
                    full_context1 = chunk['context']
            placeholder1.markdown(full_response1)
            # Citations with S3 pre-signed URL
            citations1 = extract_citations(full_context1)
            with st1.expander("Mostrar referencias >"):
                for citation in citations1:
                    st1.write("**Contenido:** ", citation.page_content)
                    s3_uri = citation.metadata['location']['s3Location']['uri']
                    bucket, key = parse_s3_uri(s3_uri)
                      #  presigned_url = create_presigned_url(bucket, key)
                     #   if presigned_url:
                     #       st.markdown(f"**Fuente:** [{s3_uri}]({presigned_url})")
                     #   else:
                      #  st.write(f"**Fuente**: {s3_uri} (Presigned URL generation failed)")
                    st1.write(f"**Fuente**: *{key}* ")
       
                    st1.write("**Score**:", citation.metadata['score'])
                    st1.write("--------------")

            # session_state append
            #st1.session_state.messages1.append({"role": "assistant", "content": full_response1})


             #session_state con referencias
            st1.session_state.messages1.append({
            "role": "assistant",
            "content": full_response1,
            "citations": citations1  # Guardar referencias junto con la respuesta.
            })
