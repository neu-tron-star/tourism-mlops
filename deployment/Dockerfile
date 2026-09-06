# Docker configuration retained for the assignment's Dockerfile requirement.
# Hugging Face currently gates Docker Spaces on paid plans; the free deployment
# used by this project is the Gradio Space defined in deployment/app.py.
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

EXPOSE 7860

CMD ["streamlit", "run", "streamlit_app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.headless=true", "--server.enableXsrfProtection=false"]
