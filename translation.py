import string
import hashlib
import requests
import random
from urllib import parse


class BaiDuTranslate1:

    def __init__(self,appid,key,fromLan='auto',toLan='zh'):
        self.appid = appid # 百度翻译api控制台的APP ID,申领地址：'https://api.fanyi.baidu.com/api/trans/product/desktop?req=developer'
        self.key = key  # 百度翻译api控制台的密钥
        self.fromLan = fromLan # 自动识别原语言
        self.toLan = toLan    # 目标语言默认为中文
        '''
        原语言可选:
        yue 粤语
        en 英语
        th 泰语
        希腊语 el
        fra 法语
        spa 西班牙语
        ara 俄语
        est 爱沙尼亚语
        更多见官方文档:'https://fanyi-api.baidu.com/product/113'
        '''

    # md5加密
    def md5Encryption(self,text):
        hashl = hashlib.md5()
        hashl.update(text.encode(encoding='utf8'))
        secret_key = hashl.hexdigest()
        return secret_key

    # 判断字符串是否存在汉字，根据需要是否调用
    def IsContainChinese(self,check_str):
        for ch in check_str:
            if u'\u4e00' <= ch <= u'\u9fff':
                return True
        return False

    #result需传参为空列表，指定目标value的key，这里是'dst'
    def get_key(self,js_data, target_key, results=[]):
        if isinstance(js_data, dict):
            for key in js_data.keys():
                data = js_data[key]
                self.get_key(data, target_key, results=results)
                if key == target_key:
                    results.append(data)
        elif isinstance(js_data, list) or isinstance(js_data, tuple):
            for data in js_data:
                self.get_key(data, target_key, results=results)
        return results

    # 返回10个随机码
    def create_alt(self):
        alt = random.choices(string.ascii_letters,weights=None,cum_weights=None,k=10)
        return ''.join(alt)

    def create_sign(self,q):
        alt = self.create_alt()
        str_= self.appid + q + alt + self.key
        sign = self.md5Encryption(str_)
        return alt,sign

    # 请求api，返回翻译结果
    def requestApi(self,q):
        translateApi = 'https://fanyi-api.baidu.com/api/trans/vip/translate?'
        salt,sign = self.create_sign(q)
        url = translateApi + 'q=' + parse.quote(q,encoding='utf-8') + '&from=' + self.fromLan +'&to='+ self.toLan +'&appid=' + appid + '&salt=' + salt + '&sign=' + sign
        re = requests.get(url)
        data = re.json()
        result = ''.join(self.get_key(js_data=data, target_key='dst', results=[]))
        return result
class BaiDuTranslate2:

    def __init__(self,appid,key,fromLan='auto',toLan='en'):
        self.appid = appid # 百度翻译api控制台的APP ID,申领地址：'https://api.fanyi.baidu.com/api/trans/product/desktop?req=developer'
        self.key = key  # 百度翻译api控制台的密钥
        self.fromLan = fromLan # 自动识别原语言
        self.toLan = toLan    # 目标语言默认为中文
        '''
        原语言可选:
        yue 粤语
        en 英语
        th 泰语
        希腊语 el
        fra 法语
        spa 西班牙语
        ara 俄语
        est 爱沙尼亚语
        更多见官方文档:'https://fanyi-api.baidu.com/product/113'
        '''

    # md5加密
    def md5Encryption(self,text):
        hashl = hashlib.md5()
        hashl.update(text.encode(encoding='utf8'))
        secret_key = hashl.hexdigest()
        return secret_key

    # 判断字符串是否存在汉字，根据需要是否调用
    def IsContainChinese(self,check_str):
        for ch in check_str:
            if u'\u4e00' <= ch <= u'\u9fff':
                return True
        return False

    #result需传参为空列表，指定目标value的key，这里是'dst'
    def get_key(self,js_data, target_key, results=[]):
        if isinstance(js_data, dict):
            for key in js_data.keys():
                data = js_data[key]
                self.get_key(data, target_key, results=results)
                if key == target_key:
                    results.append(data)
        elif isinstance(js_data, list) or isinstance(js_data, tuple):
            for data in js_data:
                self.get_key(data, target_key, results=results)
        return results

    # 返回10个随机码
    def create_alt(self):
        alt = random.choices(string.ascii_letters,weights=None,cum_weights=None,k=10)
        return ''.join(alt)

    def create_sign(self,q):
        alt = self.create_alt()
        str_= self.appid + q + alt + self.key
        sign = self.md5Encryption(str_)
        return alt,sign

    # 请求api，返回翻译结果
    def requestApi(self,q):
        translateApi = 'https://fanyi-api.baidu.com/api/trans/vip/translate?'
        salt,sign = self.create_sign(q)
        url = translateApi + 'q=' + parse.quote(q,encoding='utf-8') + '&from=' + self.fromLan +'&to='+ self.toLan +'&appid=' + appid + '&salt=' + salt + '&sign=' + sign
        re = requests.get(url)
        data = re.json()
        result = ''.join(self.get_key(js_data=data, target_key='dst', results=[]))
        return result
# 使用说明,多次请求要加上间隔,否则请求频繁得不到结果
# eg:
q = 'test' # 要翻译的内容
key = 'h9yqjddXVTuaoIIu1Tt8' # 你自己的密钥
appid = '20220706001266086' # 你自己的appid
# 翻译结果,可根据需要在 BaiDuTranslate()里加上fromLan,toLan指定翻译前后语言
# 如 BaiDuTranslate(key=key,appid=appid,fromLan='en',toLan='fra')
# q = BaiDuTranslate1(key=key,appid=appid).requestApi(q=q)
# translate_result = BaiDuTranslate2(key=key,appid=appid).requestApi(q=q)
# print(translate_result)
import pandas as  pd
df = pd.read_csv("feedback-prize-effectiveness/train.csv")
for i in range(df.shape[0]):
    t = BaiDuTranslate1(key=key, appid=appid).requestApi(q=df['discourse_text'][i])
    df['discourse_text'][i]=t
df.to_csv('cn_output.csv', index=False)
