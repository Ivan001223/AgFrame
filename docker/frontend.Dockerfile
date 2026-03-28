FROM node:22-alpine

WORKDIR /app

ARG NEXT_PUBLIC_API_URL=http://127.0.0.1:8000

ENV NODE_ENV=production \
    NEXT_TELEMETRY_DISABLED=1 \
    NEXT_PUBLIC_API_URL=${NEXT_PUBLIC_API_URL}

COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

COPY frontend ./

RUN npm run build

EXPOSE 3000

CMD ["npm", "run", "start", "--", "--hostname", "0.0.0.0", "--port", "3000"]
