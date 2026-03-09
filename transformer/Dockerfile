# Use a slim Python base
FROM public.ecr.aws/docker/library/python:3.12-slim

# 1. Install Lambda Web Adapter (This handles the streaming magic)
COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:0.8.4 /lambda-adapter /opt/extensions/lambda-adapter

# 2. Tell the adapter to use Streaming Mode
ENV AWS_LWA_INVOKE_MODE=RESPONSE_STREAM
ENV PORT=8000

WORKDIR /var/task

# 3. Install dependencies (Include uvicorn, numpy, fastapi)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy your app
COPY app/ ./app/

# 5. Run with Uvicorn (NOT Mangum)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]