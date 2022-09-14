from googletrans import Translator
# 设置Google翻译服务地址
translator = Translator(service_urls=[
      'translate.google.cn'
])

# translation = translator.translate('I think that the face is a natural landform because there is no life on Mars that we have descovered yet', dest='zh-CN')
# print(translation.text)
import time
import pandas as  pd
df = pd.read_csv("en_output.csv")
i=37664
while(i<df.shape[0]):
    i=i+1
    while(1):

        try:
            translation = translator.translate(df['discourse_text'][i], dest='zh-CN')
            break
        except:
            time.sleep(0.1)
    q=translation.text
    # print(i,translation.text+"\n")
    translation.text=''
    translation.pronunciation=''
    while(1):

        try:
            translation = translator.translate(q, dest='EN')
            break
        except:
            time.sleep(0.1)

    df['discourse_text'][i]=translation.text
    print(i,translation.text+"\n")
    translation.text=''
    translation.pronunciation=''
    df.to_csv('en_output.csv', index=False)

