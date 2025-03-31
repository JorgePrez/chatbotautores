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
import streamlit.components.v1 as components
import random
    


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
          #  st.error("AWS credentials not available")
            return ""
        return response

def parse_s3_uri(uri: str) -> tuple:
        """Parse S3 URI to extract bucket and key"""
        parts = uri.replace("s3://", "").split("/")
        bucket = parts[0]
        key = "/".join(parts[1:])
        return bucket, key