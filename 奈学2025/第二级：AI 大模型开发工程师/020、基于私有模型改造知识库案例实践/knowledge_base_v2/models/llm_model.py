
from langchain.chat_models import ChatOpenAI

from config import keys
from config.keys import Keys
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
import os

from models.chatglm3 import chatglm3


# 核心关注点temperature=0
# 对于知识库我们要求内容要严谨，不可随意发挥
def get_openai_model():
    llm_model = ChatOpenAI(openai_api_key=Keys.OPENAI_API_KEY,
                           model_name=Keys.MODEL_NAME,
                           openai_api_base=Keys.OPENAI_API_BASE,
                           temperature=0)
    return llm_model


def get_openaiEmbedding_model():
    return OpenAIEmbeddings(openai_api_key=Keys.OPENAI_API_KEY,
                            openai_api_base=Keys.OPENAI_API_BASE)



"""
私有化模型部署方案
"""

# 私有嵌入模型部署
#  "text2vec3": "shibing624/text2vec-base-chinese"
embedding_model_dict = {
    "ernie-tiny": "nghuyong/ernie-3.0-nano-zh",
    "ernie-base": "nghuyong/ernie-3.0-base-zh",
    "text2vec": "GanymedeNil/text2vec-large-chinese",
    "text2vec2": "uer/sbert-base-chinese-nli",
    "text2vec3": "/root/embedding",
}

def get_embedding_model(model_name="text2vec3"):
    """
    加载embedding模型
    :param model_name:
    :return:
    """
    encode_kwargs = {"normalize_embeddings": False}
    model_kwargs = {"device": "cuda:0"}
    print(embedding_model_dict[model_name])
    return HuggingFaceEmbeddings(
        model_name=embedding_model_dict[model_name],
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

def get_chatglm3_6b_model(model_path=keys.Keys.CHATGLM3_MODEL_PATH):
    MODEL_PATH = os.environ.get('MODEL_PATH', model_path)
    llm = chatglm3()
    llm.load_model(model_name_or_path=MODEL_PATH)
    return llm