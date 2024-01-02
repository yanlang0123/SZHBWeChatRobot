import inspect
import traceback
from copy import deepcopy
from pprint import pformat
from types import GenericAlias
from typing import get_origin, Annotated
import json
import requests
import random
import time
from tool.comfyUI_api import ComfyUIApi
import subprocess
import asyncio
from wcferry import Wcf
from win32api import ShellExecute

_TOOL_HOOKS = {}
_TOOL_DESCRIPTIONS = {}
_TOOL_QW_DESCRIPTIONS = []


def register_tool(func: callable):
    tool_name = func.__name__
    tool_description = inspect.getdoc(func).strip()
    python_params = inspect.signature(func).parameters
    tool_params = []
    for name, param in python_params.items():
        annotation = param.annotation
        if annotation is inspect.Parameter.empty:
            raise TypeError(f"Parameter `{name}` missing type annotation")
        if get_origin(annotation) != Annotated:
            raise TypeError(f"Annotation type for `{name}` must be typing.Annotated")

        typ, (description, required) = annotation.__origin__, annotation.__metadata__
        typ: str = str(typ) if isinstance(typ, GenericAlias) else typ.__name__
        if not isinstance(description, str):
            raise TypeError(f"Description for `{name}` must be a string")
        if not isinstance(required, bool):
            raise TypeError(f"Required for `{name}` must be a bool")

        tool_params.append({
            "name": name,
            "description": description,
            "type": typ,
            "required": required
        })
    tool_def = {
        "name": tool_name,
        "description": tool_description,
        "params": tool_params,
        "parameters":tool_params
    }

    #print("[registered tool] " + pformat(tool_def))
    _TOOL_HOOKS[tool_name] = func
    _TOOL_DESCRIPTIONS[tool_name] = tool_def
    _TOOL_QW_DESCRIPTIONS.append(tool_def)

    return func


def dispatch_tool(tool_name: str, tool_params: dict) -> str:
    if tool_name not in _TOOL_HOOKS:
        return f"Tool `{tool_name}` not found. Please use a provided tool."
    tool_call = _TOOL_HOOKS[tool_name]
    try:
        if len(tool_params) > 0:
            ret = tool_call(**tool_params)
        else:
            ret = tool_call()

    except:
        ret = traceback.format_exc()
    return ret


def get_tools() -> dict:
    return deepcopy(_TOOL_DESCRIPTIONS)

def get_qw_tools() -> list:
    return deepcopy(_TOOL_QW_DESCRIPTIONS)


# Tool Definitions

@register_tool
def random_number_generator(
        seed: Annotated[int, 'The random seed used by the generator', True],
        range: Annotated[tuple[int, int], 'The range of the generated numbers', True],
) -> int:
    """
    Generates a random number x, s.t. range[0] <= x < range[1]
    """
    if not isinstance(seed, int):
        raise TypeError("Seed must be an integer")
    if not isinstance(range, tuple) and not isinstance(range, list):
        raise TypeError("Range must be a tuple")
    if not isinstance(range[0], int) or not isinstance(range[1], int):
        raise TypeError("Range must be a tuple of integers")

    import random
    return random.Random(seed).randint(*range)


@register_tool
def get_weather(
        city_name: Annotated[str, 'The name of the city to be queried', True],
) -> str:
    """
    Get the current weather for `city_name`
    """

    if not isinstance(city_name, str):
        raise TypeError("City name must be a string")

    key_selection = {
        "current_condition": ["temp_C", "FeelsLikeC", "humidity", "weatherDesc", "observation_time"],
    }
    import requests
    try:
        resp = requests.get(f"https://wttr.in/{city_name}?format=j1")
        resp.raise_for_status()
        resp = resp.json()
        ret = {k: {_v: resp[k][0][_v] for _v in v} for k, v in key_selection.items()}
    except:
        import traceback
        ret = "Error encountered while fetching weather data!\n" + traceback.format_exc()

    return str(ret)

wcf = Wcf(debug=True)

@register_tool
def send_wechat_message(
        message: Annotated[str, '消息内容', True],
        to_user: Annotated[str, '消息接收人', True]):
    '''
    发送微信消息给`to_user`消息内容为`message`
    '''
    try:
        friends=wcf.get_friends()
        for friend in friends:
            if friend['name'] == to_user:
                if wcf.send_text(message,friend['wxid'])==0:
                    answer = {"success": True,"res": "发送成功","res_type": "text"}
                    return answer
                
        answer = {"success": False,"res": "发送失败" ,"res_type": "text"}
        return answer
    except:
        answer = {"success": False,"res": "发送异常" ,"res_type": "text"}
        return answer
            
    
    
softwares={
    "微信": r"C:\Program Files (x86)\Tencent\WeChat\WeChat.exe",
    "浏览器": r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    "开发工具": r"I:\Programs\Microsoft VS Code\Code.exe",
    "绘图工具": r"F:\ComfyUI\启动.bat",
    "记事本": "notepad",
    "计算器": "calc",
    "控制面版": "control",
    "画图": "mspaint",
}
#启动软件
@register_tool
def start_software(software_name: Annotated[str, '软件名称 e.g. 微信，浏览器，开发工具，绘图工具，记事本，计算器，控制面版，画图', True]) -> dict:
    '''
    启动或打开`software_name`软件
    '''
    # 创建事件循环并运行异步任务
    try:
        ShellExecute(None, "open", softwares[software_name], None, None, 1)
    except:
        answer = {"success": False,"res": "启动失败" ,"res_type": "text"}
        return answer
            
    answer = {"success": True,"res": "启动成功" ,"res_type": "text"}
    return answer

@register_tool
def clear_chat_record() -> dict:
    '''清除聊天纪录'''
    answer = {"success": True,"res": "清除成功" ,"res_type": "text"}
    return answer


@register_tool
def get_confyui_image(prompt: Annotated[str, '要生成图片的英文提示词', True],
                      style_name: Annotated[str, '风格 e.g. 漫画，抽象，水墨画，滴墨，未来派', False]=None) -> dict:
    '''
    生成图片`prompt`提示词和风格`style_name`的图片
    '''
    styles=''
    if style_name:
        f = open("Q:\\ChatGLM3-main\\tool\\styles.json",'r', encoding='utf-8')
        data = json.load(f)
        f.close()
        if style_name in data:
            styles=data[style_name]
        else:
            for k,v in data.items():
                if k.find(style_name)>-1:
                    styles=v
                    break

    
    with open("Q:\\ChatGLM3-main\\tool\\base.json", "r", encoding="utf-8") as f:
        data2 = json.load(f)
        data2['prompt']['3']['inputs']['seed']=''.join(random.sample('123456789012345678901234567890',14))
        data2['prompt']['4']['inputs']['ckpt_name']='juggernautXL_version6Rundiffusion.safetensors'
        data2['prompt']['17']['inputs']['text_trans']=prompt
        data2['prompt']['15']['inputs']['button']=styles
        #data2['prompt']['19']['inputs']['output_path']="./output/tdfcs"
        #data2['prompt']['19']['inputs']['filename_prefix']='TTTDFCS_0'
        cfui=ComfyUIApi()
        images,imgs = cfui.get_images(data2['prompt'],isUrl=True)
        if len(imgs)>0:
            try:
                ShellExecute(None, "open", 'F:/ComfyUI/'+imgs[0]['type']+'/'+imgs[0]['filename'], None, None, 1)
            except:
                print("打开异常")
        return {'res':images[0],'res_type':'image'}

if __name__ == "__main__":
    print(dispatch_tool("get_weather", {"city_name": "beijing"}))
    print(get_tools())
