FROM nginx:alpine

WORKDIR /usr/share/nginx/html

# Удаляем дефолтную страницу nginx
RUN rm -rf ./*

# Копируем статику
COPY index.html .
COPY style.css .
COPY app.js .

# Копируем конфиг nginx
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80
