import os, re, json
import telebot
from telebot.types import InputMediaPhoto

import recommendation_clothes

TEMP_PATH = './temp_recommendation/'

df = recommendation_clothes.init_data()
embeddings = recommendation_clothes.get_embeddings()

with open('./bot_answers.json', encoding='utf-8') as strings_file:
    BOT_ANSWERS = json.load(strings_file)

with open('./token.json') as token_file:
    token = json.load(token_file)['API']['TOKEN']

bot = telebot.TeleBot(token)

@bot.message_handler(commands=['start'])
def start(message):
    reply = BOT_ANSWERS['start']
    bot.send_message(message.chat.id, reply)

@bot.message_handler(content_types=['text'])
def handle_text(message):
    if re.match(r'\D', message.text, re.A):
        reply =  BOT_ANSWERS['not_valid']
        bot.send_message(message.chat.id, reply)
        return

    try:
        index = int(message.text)
        print('message.text = ', message.text)
    except BaseException: pass

    flag = False
    try:
        flag = recommendation_clothes.recommendation(index, df, embeddings)
    except BaseException as e:
        print(e)

    if flag:
        reply = BOT_ANSWERS['submission']
        bot.send_photo(message.chat.id, open('original.jpg', 'rb'), caption=reply)

        media_group = []
        print(os.listdir(TEMP_PATH))
        for i, name in enumerate(os.listdir(TEMP_PATH)):
            try:
                f = open(f'{TEMP_PATH}{name}', 'rb')
              
                media_group.append(InputMediaPhoto(f))
            except BaseException:
                pass
        print(media_group)
        reply = BOT_ANSWERS['recommendation']
        bot.send_message(message.chat.id, reply)
        bot.send_media_group(message.chat.id, media=media_group)
    else: 
        reply =  BOT_ANSWERS['aborted']
        bot.send_message(message.chat.id, reply)

bot.polling(none_stop=True, interval=0)
