import json
from aiohttp import web
# from mainpage import textarea_save_ajax

#df = pd.read_csv('templates/100.csv')
#print(df.head())

def mainpage(CORE, NLP):
    CORE.debug('/components/mainpage/mainpage.py')

    CORE.addHeadFile('/templates/general/css/DAN.css')
    CORE.addHeadFile('/templates/general/js/DAN.js')
    CORE.addHeadFile('/templates/general/css/mainpage.css')
    CORE.addHeadFile('/templates/general/js/mainpage.js')

    CORE.content = f'''
        <div class="dan_flex_row_start">
            <div class="mp_panel">
                <div class="mp_panel_title">Загрузка данных</div>
                <div>
                    <textarea id="textarea" class="dan_input" placeholder="Введите текст пресс релиза"></textarea>
                    <input id="button_submit" class="dan_button_red" name="submit" type="submit" value="Отправить">
                    <p class="file-return"></p>
                </div>
            </div>
            <div class="mp_panel">
                <div class="mp_panel_title">Информация</div>
                <div id="rating"></div>
            </div>
        </div>
    '''
