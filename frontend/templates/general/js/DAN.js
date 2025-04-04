let DAN = {}

DAN.$ = function(id){return document.getElementById(id)}

DAN.ajax = function(url, form = false, callback){
	let req = new XMLHttpRequest()

	req.upload.onprogress = function(event) {
		// console.log(event.loaded + ' / ' + event.total)
		if (DAN.$('dan_modal_progress')) {
			let result = Math.round((event.loaded / event.total)*100)
			DAN.$('dan_modal_progress').value = result
			if (DAN.$('dan_modal_progress_html'))
				DAN.$('dan_modal_progress_html').innerHTML = result
		}
	}

	req.onreadystatechange = ()=>{
		if (req.readyState == 4 && req.status == 200) {
			console.log(req.responseText);
			let data = JSON.parse(req.responseText)
			if (data.answer == 'success')
				callback(data)
			else if(data.answer == 'error')
				DAN.modal.add(data.message)
			else
				DAN.errLog('Ошибка DAN.ajax => ' + req.responseText)
		}
	}

	req.open('post', url, true);
	req.send(form)		
}


DAN.modal = {
	block_up: false, // Не блокировать модальное окно

	// Добавить блок
	add(content_html, width = 600, height = 100){
		if (document.getElementById('dan_modal_black') !== null) 
			DAN.modal.del()

		let modal_black = document.createElement('div')
		modal_black.id = 'dan_modal_black'
		document.body.insertBefore(modal_black, document.body.children[0]) // черный слой

		let cross = document.createElement('div')
		cross.id = 'dan_modal_cross'
		cross.innerHTML = '<svg><use xlink:href="/templates/general/images/sprite.svg#delete"></use></svg>'

		let content = document.createElement('div')
		content.id = 'dan_modal_content'
		content.innerHTML = content_html

		let modal_white = document.createElement('div')
		modal_white.id = 'dan_modal_white'
		modal_white.style.maxWidth = width + 'px'
		modal_white.style.minHeight = height + 'px'
		modal_white.appendChild(cross)		
		modal_white.appendChild(content)
		modal_black.appendChild(modal_white)

		modal_white.onmousedown = (e)=>{
			e.stopPropagation()
		}

		modal_black.onmousedown = DAN.modal.del
		cross.onmousedown = DAN.modal.del
	},

	// Удалить блок
	del() {
		if (!DAN.$('dan_modal_white')) 
			return
		if (!this.block_up)
			document.body.removeChild(DAN.$('dan_modal_black'))
	},

	// Блокирует удаление модального окна
	block(b) {
		this.block_up = b
	},

	// Спиннер
	spinner(text=false) {
		if (!DAN.$('dan_modal_content')) 
			DAN.modal.add('', 200, 200)
		text_html = text ? '<h1 class="dan_spinner_h1">' + text + '</h1>' : ''
		DAN.$('dan_modal_content').innerHTML = text_html +
		'<div class="dan_spinner_wrap">' + 
			'<svg id="dan_spinner" xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24"><path d="M11.501 4.025v-4.025h1v4.025l-.5-.025-.5.025zm-7.079 5.428l-3.884-1.041-.26.966 3.881 1.04c.067-.331.157-.651.263-.965zm5.995-5.295l-1.039-3.878-.967.259 1.041 3.883c.315-.106.635-.197.965-.264zm-6.416 7.842l.025-.499h-4.026v1h4.026l-.025-.501zm2.713-5.993l-2.846-2.845-.707.707 2.846 2.846c.221-.251.457-.487.707-.708zm-1.377 1.569l-3.48-2.009-.5.866 3.484 2.012c.15-.299.312-.591.496-.869zm13.696.607l3.465-2-.207-.36-3.474 2.005.216.355zm.751 1.993l3.873-1.038-.129-.483-3.869 1.037.125.484zm-3.677-5.032l2.005-3.472-.217-.125-2.002 3.467.214.13zm-1.955-.843l1.037-3.871-.16-.043-1.038 3.873.161.041zm3.619 2.168l2.835-2.834-.236-.236-2.834 2.833.235.237zm-9.327-1.627l-2.011-3.484-.865.5 2.009 3.479c.276-.184.568-.346.867-.495zm-4.285 8.743l-3.88 1.04.26.966 3.884-1.041c-.106-.314-.197-.634-.264-.965zm11.435 5.556l2.01 3.481.793-.458-2.008-3.478c-.255.167-.522.316-.795.455zm3.135-2.823l3.477 2.007.375-.649-3.476-2.007c-.116.224-.242.439-.376.649zm-1.38 1.62l2.842 2.842.59-.589-2.843-2.842c-.187.207-.383.403-.589.589zm2.288-3.546l3.869 1.037.172-.644-3.874-1.038c-.049.218-.102.434-.167.645zm.349-2.682l.015.29-.015.293h4.014v-.583h-4.014zm-6.402 8.132l1.039 3.879.967-.259-1.041-3.884c-.315.106-.635.197-.965.264zm-1.583.158l-.5-.025v4.025h1v-4.025l-.5.025zm-5.992-2.712l-2.847 2.846.707.707 2.847-2.847c-.25-.22-.487-.456-.707-.706zm-1.165-1.73l-3.485 2.012.5.866 3.48-2.009c-.185-.278-.347-.57-.495-.869zm2.734 3.106l-2.01 3.481.865.5 2.013-3.486c-.299-.149-.591-.311-.868-.495zm1.876.915l-1.042 3.886.967.259 1.04-3.881c-.33-.067-.65-.158-.965-.264z"/></svg>' + 
		'</div>'
	},

	update(content) {
		DAN.$('dan_modal_content').innerHTML = content
	}
}


DAN.show = function(_class, _id){
	if (!DAN.$(_id))
		return

	let obj = new DAN_show(_class, _id)

	// Запуск при повторной инициализации
	if (document.readyState)
		run (obj)

	// Запуск при загрузке
	document.addEventListener('DOMContentLoaded', function(){
		run (obj)
	})

	function run() {
		for (let i = 0; i < obj.sum; i++) {
			obj.img_arr[i].style.cursor = 'url(/templates/images/lupa.png), auto'

			obj.img_arr[i].onclick = (e)=>{
				// Создаём модальное окно

				if(!obj.bg){
					// Чёрный фон
					obj.bg = document.createElement('div')
					obj.bg.id = 'dan_show_black'
					document.body.insertBefore(obj.bg, document.body.children[0])

					obj.bg.onclick = function(){
						obj.del()
						obj.bg = false
						obj.image = false
					}
				}


				obj.output(e.target)
		
				obj.image.onclick = function(e){		
					e.stopPropagation()
				}				

				DAN.$('dan_show_nav_left').onclick = (e)=>{
					e.stopPropagation()
					obj.stop()
					obj.prev()
				}
				
				DAN.$('dan_show_nav_play').onclick = (e)=>{
					e.stopPropagation()				
			
					play('button')

					function play (_but = false){ // _but == true -> анимация запущена кнопкой, false - автоматом по циклу
						// Кнопка play / stop
						if(_but == 'button'){
							if(!obj.timer){
								DAN.$('dan_show_nav_play').removeChild(DAN.$('dan_show_nav_play').lastChild)
								DAN.$('dan_show_nav_play').insertAdjacentHTML('afterbegin', '<svg class="dan_show_nav"><use xmlns:xlink="http://www.w3.org/1999/xlink" xlink:href="/templates/general/images/sprite.svg#pause"></use></svg>')
							}
							else{						
								obj.stop()
								return
							}
						}

						obj.next()
						obj.timer = setTimeout(function(){
							requestAnimationFrame(play)
						}, obj.interval)
					}				
				}
				
				DAN.$('dan_show_nav_right').onclick = (e)=>{
					e.stopPropagation()
					obj.stop()
					obj.next()
				}	
			}
		}
	}
}


// Класс для DAN.show()
class DAN_show{
	constructor(_class, _id){
		this.container = DAN.$(_id) // Контейнет
		this.img_arr = this.container.getElementsByClassName(_class) // Массив изображений в контейнере
		this.sum = this.img_arr.length // Количество изображений
		this.bg = false // Фон с затемнением
		this.wrap = false  // Оболочка
		this.image = false // Текущее большое изображение
		this.interval = 2000 // Интервал анимации
		this.nav_play = false // Кнопка навигации
		this.num = false // Текущий индекс массива изображений
		this.timer = false // Номер таймера
	}

	output(img){
		// Изображение
		let img_out = document.createElement('img')
		img_out.src = img.src
		img_out.id = 'dan_show_image'

		for (let i = 0; i < this.sum; i++) {
		    if (typeof this.img_arr[i] == 'undefined')
		        continue

			if (img.src == this.img_arr[i].src)
				this.num = i
		}

		// Обёртка
		this.wrap = document.createElement('div')
		this.wrap.id = 'dan_show_wrap'
		this.bg.insertAdjacentElement('afterbegin', this.wrap)

		// Изображение
		this.image = this.wrap.insertAdjacentElement('afterbegin', img_out)

		// Навигация
		let cross = document.createElement('div')
		cross.id = 'dan_show_cross'
		cross.innerHTML = '<svg><use xlink:href="/templates/general/images/sprite.svg#delete"></use></svg>'
		this.wrap.insertAdjacentElement('afterbegin', cross)

		let nav_left = document.createElement('div')
		nav_left.id = 'dan_show_nav_left'
		this.wrap.insertAdjacentElement('afterbegin', nav_left)
		nav_left.insertAdjacentHTML('afterbegin', '<svg class="dan_show_nav"><use xmlns:xlink="http://www.w3.org/1999/xlink" xlink:href="/templates/general/images/sprite.svg#prev"></use></svg>')

		let nav_right = document.createElement('div')
		nav_right.id = 'dan_show_nav_right'
		this.wrap.insertAdjacentElement('beforeend', nav_right)
		nav_right.insertAdjacentHTML('afterbegin', '<svg class="dan_show_nav"><use xmlns:xlink="http://www.w3.org/1999/xlink" xlink:href="/templates/general/images/sprite.svg#next"></use></svg>')

		let nav_play = document.createElement('div')
		nav_play.id = 'dan_show_nav_play'
		this.wrap.insertAdjacentElement('afterbegin', nav_play)
		nav_play.insertAdjacentHTML('afterbegin', '<svg class="dan_show_nav"><use xmlns:xlink="http://www.w3.org/1999/xlink" xlink:href="/templates/general/images/sprite.svg#play"></use></svg>')		
	}
	
	prev(){
		this.num--
		if(this.num < 0) this.num = this.img_arr.length - 1
		this.image.src = this.img_arr[this.num].src
	}

	next(){	
		this.num++
		if(this.num > (this.sum - 1)) this.num = 0
		this.image.src = this.img_arr[this.num].src		
	}

	stop(){
		DAN.$('dan_show_nav_play').removeChild(DAN.$('dan_show_nav_play').lastChild)
		DAN.$('dan_show_nav_play').insertAdjacentHTML('afterbegin', '<svg class="dan_show_nav"><use xmlns:xlink="http://www.w3.org/1999/xlink" xlink:href="/templates/general/images/sprite.svg#play"></use></svg>')		
		clearTimeout(this.timer)
		this.timer = false
	}

	del(e){
		this.stop()
		document.body.removeChild(DAN.$('dan_show_black'))
	}
}


// Прокрутка к _id с временным интервалом _duration и отступом сверху _offset
DAN.jumpTo = function(_id = null, _duration = 300, _offset = 0){
	let timeInterval = 16.667
	let element = DAN.$(_id)

	if (!element)
		element = document.body

	_duration = _duration > 0 ? _duration : 0
	_offset = parseInt(_offset)

	let elementPosY = 0

	while(element){
		elementPosY += parseFloat(element.offsetTop)
		element = element.offsetParent
	}

	let step = _duration/timeInterval

	if (_duration == 0){
		window.scrollTo(0, elementPosY)
		return
	}

	let scrolled = window.pageYOffset || document.documentElement.scrollTop

	let i = 0

	function animate(fraction) {

		requestAnimationFrame(function animate(fraction) {

			fraction = i/step
			if (fraction > 1) 
				fraction = 1
			let fract = (Math.sin(fraction*Math.PI - Math.PI/2) + 1)/2
			let scroll = scrolled + (elementPosY - scrolled + _offset)*fract

			window.scrollTo(0, scroll)
			i++

			if (fraction < 1) 
				requestAnimationFrame(animate)
		})
	}

	animate(0)
}


// Анимация появления
DAN.appearance = function(id, classCss) {
	let object = document.getElementById(id);

	if(!object)
		return;

	let callback = function() {
		let objectY = object.offsetTop;
		let scrollY = window.pageYOffset || document.documentElement.scrollTop;
		let windowBottom = scrollY + window.innerHeight;

		if(scrollY < objectY && objectY < windowBottom)
			object.classList.add(_classCss);
		else
			object.classList.remove(_classCss);

	};

	window.addEventListener("scroll", callback);
	callback();
};


DAN.errLog = function(text) {
	console.log(text)
}