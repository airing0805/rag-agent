# LangChain 的 ArxivQueryRun 工具
from pydantic import BaseModel, Field
from langchain_community.utilities import ArxivAPIWrapper

"""
langChain Docs:https://python.langchain.com/v0.2/docs/integrations/tools/arxiv/
"""

from pydantic import BaseModel, Field
from langchain_core.tools import tool

import requests
import json

open_weather_key = "5c939a7cc59eb8696f4cd77bf75c5a9a"

def get_weather(loc):
    """
    查询即时天气函数
    :param loc: 必要参数，字符串类型，用于表示查询天气的具体城市名称，\
    注意，中国的城市需要用对应城市的英文名称代替，例如如果需要查询上海市天气，则loc参数需要输入'ShangHai'；
    :return：OpenWeather API查询即时天气的结果，具体URL请求地址为：https://api.openweathermap.org/data/2.5/weather\
    返回结果对象类型为解析之后的JSON格式对象，并用字符串形式进行表示，其中包含了全部重要的天气信息
    """
    # Step 1.构建请求
    url = "https://api.openweathermap.org/data/2.5/weather"

    # Step 2.设置查询参数
    params = {
        "q": loc,
        "appid": open_weather_key,  # 输入API key
        "units": "metric",  # 使用摄氏度而不是华氏度
        "lang": "zh_cn"  # 输出语言为简体中文
    }

    # Step 3.发送GET请求
    response = requests.get(url, params=params)

    # Step 4.解析响应
    data = response.json()
    return json.dumps(data)

class GetWeatherInput(BaseModel):
    query: str = Field(description="输入需要查询天气的城市地址")


if __name__ == "__main__":
    docs = get_weather("ShangHai")
    print(docs)