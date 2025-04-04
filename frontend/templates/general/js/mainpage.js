window.addEventListener('DOMContentLoaded', function(){
	IMPORT_DATA.init();
});


IMPORT_DATA = {
    // Инициализация
    init() {       
        DAN.$('button_submit').onclick = IMPORT_DATA.send_form;
    },


    // Сохранение настроек
    send_form() {
        let text = DAN.$('textarea').value;
        if (text.length < 1000) {
            alert ('Текст слишком короткий. Текст должен быть не менее 100 символов');
            return;
        }

        let form = new FormData();
		form.append('text', text)	
	
		DAN.ajax('/text_send_ajax', form, function(data) {
            DAN.$('rating').innerHTML = '<div class="rating">Рейтинг: ' + data.rating + '</div>';
		})
    }
}