import time
from datetime import datetime
import config

def auth(CORE):
    current_datetime = datetime.now()

    CORE.debug(f'===== AUTH ===== {current_datetime}')
    
    # Статус авторизации - 1, если есть сессия
    if 'auth' in CORE.session:
        CORE.debug(f'Авторизация - успешна, {CORE.session}')
        CORE.auth = 1
        return True

    # Если получаем данные с формы входа
    if (CORE.p[0] == 'auth'):
        if (CORE.request.method == 'POST'):
            CORE.debug(f'Проверка пароля {CORE.post}')
            if ('login' in CORE.post and 
                'password' in CORE.post and
                'button' in CORE.post and
                CORE.post['login'] == config.admin_login and 
                CORE.post['password'] == config.admin_pass):
                CORE.debug('Проверка пароля прошла успешно!')
                CORE.session['auth'] = time.time()
                CORE.auth = 1
                return True

    if not CORE.auth:
        # Нет авторизации - выводим фому
        CORE.addHeadFile('/templates/general/css/DAN.css')
        CORE.addHeadFile('/templates/general/css/auth.css')

        CORE.title = 'Авторизация'
        CORE.content = '''<form method="post" action="/auth">
        <div class="login_form_container">
            <div class="login_form_text">Логин (admin)</div>
            <div><input name="login" class="dan_input" value=""></div>
            <div class="login_form_text">Пароль</div>
            <div><input name="password" class="dan_input" type="password" value=""></div>
            <div><input name="button" class="dan_input login_form_submit" type="submit" value="Вход"></div>
        </div>
        </form>
        '''