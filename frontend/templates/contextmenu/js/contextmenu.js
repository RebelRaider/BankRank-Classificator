// class_name - класс, для которого вызывается контекстное меню
// contextmenu_content - двумерный массив, в 0 индексе стоит url или #функция, которая вставиться так onClick = function(objContext);
// Пример инициализации:
// class_name = "contextmenu_shopmain";
// var contextmenu_shopmain = [
// 	 ["admin/com/shop", "contextmenu_tools", "Настройки"],
//	 ["admin/com/shop/orders", "contextmenu_orders", "Заказы", "_blank"],
//	 ["#function", "contextmenu_import", "Экспорт / Импорт"],			
// ];
// contextmenu(class_name, contextmenu_shopmain);


DAN.contextmenu = {
	objContext: new Object(),
	
	add(class_name, content, type_click='right'){
		var class_context = document.getElementsByClassName(class_name);
		var class_context_length = class_context.length;
		for (var i = 0; i < class_context.length; i++){
			if(type_click.indexOf('right') != -1){
				class_context[i].oncontextmenu = (e)=>{
					DAN.contextmenu.out(e, class_name, content);
				}
			}
			if(type_click.indexOf('left') != -1){
				class_context[i].onclick = (e)=>{
					DAN.contextmenu.out(e, class_name, content);
				}
			}
		}
		
		document.onclick = DAN.contextmenu.del;
		return false;		
	},

	
	// Получаем элемент на котором отслеживаем клик
	get_element(e, class_name){
		var objParent = e.target || e.srcElement;

		while(objParent){
			if(objParent.classList.contains(class_name))
				return objParent;
			objParent = objParent.parentNode;
		}
		return false;
	},
	

	// Удаляем контекстное меню
	del(e){
		var node = document.getElementById('dan_contextmenu');
		if(node)
			document.body.removeChild(node);
	},


	out(e, class_name, content){
		e.stopPropagation();

		// получаем узел с указанным классом по которому был вызван контекстный клик
		objContext = DAN.contextmenu.get_element(e, class_name, content);

		// защита от повторного контекстного окна, при повторном контекстном клике - удаляем это контекстное меню и создаём новое.
		if(DAN.$('dan_contextmenu') != null)
			DAN.contextmenu.del(e);
	
		if(objContext.getAttribute("data-id") != null)
			var data_id = objContext.getAttribute("data-id");
		else
			var data_id = '';
	
		var c_menu = document.createElement('div');
		c_menu.id = 'dan_contextmenu';
		
		var body_child_0 = document.body.children[0];
		document.body.insertBefore(c_menu, body_child_0);
	
		c_menu.style.top = (e.pageY - 10) + 'px';
		c_menu.style.left = (e.pageX - 10) +'px';

		var contextmenu_content_length = content.length;
		var out = '';
		
		for (var c = 0; c < contextmenu_content_length; c++){
			if(content[c][0][0] == '#'){ // исполнить как функцию
				var str = content[c][0];
				var f_name = str.substr(1, str.length);
				out += '<a href="#" onClick="' + f_name + '(objContext); return false;" class="' + content[c][1] + '">' + content[c][2] + '</a>';				
			} 
			else{
				if(content[c][3])
					var trg = content[c][3];
				else
					var trg = '_self';
				out += '<a target="' + trg + '" href="' + content[c][0] + '/' + data_id + '" class="' + content[c][1] + '">' + content[c][2] + '</a>';				
			}
		}
		
		c_menu.innerHTML = out;
		
		if (e.preventDefault)
			e.preventDefault();
		else
			e.returnValue = false; // вариант IE<9:
	}	
}