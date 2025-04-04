import config


class Core:
    # Инициализация при запуске системы
    def __init__(self):
        self.config = config


    # Инициализация при открытии страницы
    def initial(self):
        print('ИНИЦИАЛИЗАЦИЯ')
        self.file = False  # Полученый файл
        self.post = False  # объект post
        self.head = ''  # Материал выводимыей в внутри тега шаблона <head>
        self.tag_title = ''  # Тег title
        self.tag_description = ''  # Метатег descripton
        self.content = ''  # Основное содержимое
        self.modules = {}  # Словарь модулей
        self.headFile = []  # Файлы для вывода в шапке шаблона
        self.headCode = ''  # Строка с кодом для вывода в <HEAD>
        self.auth = 0  # Авторизация 0 => нет; 1 - 9 => администраторы; 10 - 100 => пользователи
        self.session = False
        self.nlp = False  # Тут будем хранить экземпляр класса Nlp    


    # Обработка запроса
    def procReq(self, request):
        self.request = request  #  # request aiohttp
        self.p = request.path[1:].split('/')  # Список элементов пути
        i = len(self.p)
        while i < 7:
            self.p.append('')
            i += 1


    # Добавляет файлы для вывода в шапке шаблона
    def addHeadFile(self, path):
        if path in self.headFile:
            return
        self.headFile.append(path)


    # Выводит '<script ...>, <link ...>' в шапке HTML документа
    def getHead(self):
        out = ''
        for file in self.headFile:
            file_list = file.split('.')
            if file_list[-1] == 'js':
                out += '<script src="' + file + '"></script>'
            if file_list[-1] == 'css':
                out += '<link rel="stylesheet" href="' + file + '" />'
        out += self.headCode
        return out

    
    # Выводит текст отладки
    def debug(self, txt):
        if self.debug_on:
            print(txt)
