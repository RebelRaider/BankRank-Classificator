import datetime
import sys
import base64
from cryptography import fernet
import jinja2
import aiohttp_jinja2
from aiohttp import web
from aiohttp_session import setup, get_session, session_middleware
from aiohttp_session.cookie_storage import EncryptedCookieStorage

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.layers import Embedding, Flatten, Input, concatenate, Reshape
from tensorflow.keras.utils import plot_model, to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, Callback
from tensorflow.keras.models import load_model
import tensorflow_hub as hub
import tensorflow_text as text

sys.path.append('classes')
sys.path.append('components')
sys.path.append('components/auth')
# sys.path.append('components/users')
from Core import Core
from Nlp import Nlp
from auth import auth
from mainpage import mainpage
from text_send_ajax import text_send_ajax
#from import_data import import_data
#from stat_data import stat_data
#from predict import predict
#from promising import promising


app = web.Application(client_max_size=1024**100)

# Установка сессий
fernet_key = fernet.Fernet.generate_key()
secret_key = base64.urlsafe_b64decode(fernet_key)
setup(app, EncryptedCookieStorage(secret_key))


CORE = Core()
CORE.debug_on = True  # Выводить отладочную информацию

NLP = Nlp()
NLP.model_rat = load_model('./model/model_rat_1.h5', custom_objects={'KerasLayer': hub.KerasLayer}, compile=False)
NLP.model_rat.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=(0.00001)), metrics=['accuracy'])


@aiohttp_jinja2.template('main.html')
async def index(request):
    now = datetime.datetime.now()
    CORE.debug(f'===== INDEX ===== {now}')
    CORE.debug('REQUEST:')
    CORE.debug(request)

    CORE.initial()
    CORE.procReq(request)
    CORE.post = await request.post()  # Ждём получение файлов методом POST
    CORE.session = await get_session(request)

    # Проверка авторизации
    if not auth(CORE):
        CORE.debug('Нет авторизации')
        return {'AUTH': False, 'content': CORE.content, 'head': CORE.getHead()}

    # Редирект для url авторизации
    if CORE.p[0] == 'auth':
        return web.HTTPFound('/')

    # Вызов компонента
    functions = {
        '': mainpage.mainpage,
        'text_send_ajax': text_send_ajax.text_send_ajax,
    }

    if (CORE.p[0] not in functions):
        raise web.HTTPNotFound()

    # Вызов функции возвращает не False в случае редиректа
    r = functions[CORE.p[0]](CORE, NLP)

    # Проверка и обработка редиректа
    if r:
        if 'redirect' in r:
            # Обработка редиректа
            return web.HTTPFound(r['redirect'])
        if 'ajax' in r:
            # Обработка ajax
            return web.HTTPOk(text=r['ajax'])

    return {'AUTH': True, 'content': CORE.content, 'head': CORE.getHead()}



aiohttp_jinja2.setup(app, loader=jinja2.FileSystemLoader('templates'))
app.add_routes([
    web.static('/templates', 'templates'),
    web.get('/{url:.*}', index),  # Админка
    web.post('/{url:.*}', index),  # Админка
])

if __name__ == '__main__':
    web.run_app(app, port=9100)