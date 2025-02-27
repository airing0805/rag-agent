from langchain_community.chat_models import ChatZhipuAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

ZHIPUAI_API_KEY = "2c397e8089af28761792c52d334e92c3.fZQHXOprjyAR3Xs7"

chat = ChatZhipuAI(
    api_key=ZHIPUAI_API_KEY,
    model="glm-4",
    temperature=0.5,
)

messages = [
    SystemMessage(content="你是一位诗人"),
    AIMessage(content="你好"),
    HumanMessage(content="写一个关于AI的诗词"),
]

response = chat.invoke(messages)
print(response.content)
