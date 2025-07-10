# Этап сборки
FROM node:24.3.0-alpine AS build

# Установка рабочей директории
WORKDIR /app

# Копируем зависимости
COPY package*.json ./
RUN npm install

# Копируем проект
COPY . .

# Собираем приложение
RUN npm run build

# Этап запуска
FROM nginx:alpine

# Копируем собранный билд в nginx
COPY --from=build /app/build /usr/share/nginx/html

# Опционально: добавить кастомный конфиг, если нужен SPA fallback
# COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
