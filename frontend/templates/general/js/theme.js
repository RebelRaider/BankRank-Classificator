window.addEventListener('DOMContentLoaded', function(){
	let body = document.getElementsByTagName('body')[0];
	let dark_theme = document.getElementById('dark_theme');
	let white_theme = document.getElementById('white_theme');	
	
	
	if (dark_theme) {
		white_theme.onclick = function() {		
			white_theme.classList.add('theme_ico_active');
			dark_theme.classList.remove('theme_ico_active');
			body.classList.add('white_theme');		
			body.classList.remove('dark_theme');
			document.cookie = "dan_theme=white";			
		}
		
		dark_theme.onclick = function() {
			dark_theme.classList.add('theme_ico_active');		
			white_theme.classList.remove('theme_ico_active');	
			body.classList.add('dark_theme');		
			body.classList.remove('white_theme');
			document.cookie = "dan_theme=dark";		
		}
		
		let dan_theme_results = document.cookie.match(/dan_theme=(.+?)(;|$)/);
		
		if (dan_theme_results != null) {
			if (dan_theme_results[1] == 'dark') {
				dark_theme.classList.add('theme_ico_active');		
				white_theme.classList.remove('theme_ico_active');	
				body.classList.add('dark_theme');		
				body.classList.remove('white_theme');
				document.cookie = "dan_theme=dark";	
			} else {
				white_theme.classList.add('theme_ico_active');
				dark_theme.classList.remove('theme_ico_active');
				body.classList.add('white_theme');		
				body.classList.remove('dark_theme');
				document.cookie = "dan_theme=white";
			}
		}
	}

});