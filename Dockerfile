from docker.io/pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

RUN apt-get update && apt-get install -y curl
RUN pip install --upgrade pip

WORKDIR /app

COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt 

# Optional, build the downloaded models into the container image
# The 3 models can also be downloaded dynamically when the container is running. 
# Assume the models are already saved in the ./torch 
COPY ./torch /root/.cache/torch

COPY cog.yaml /app
COPY predict.py /app


COPY test_client.py /app
COPY test_healthcheck.py /app

#  Modify the installed Cog prediction server to use IPv6
RUN  sed -i 's/0.0.0.0/::/g' /opt/conda/lib/python3.10/site-packages/cog/server/http.py

# Run the Cog prediction server 
CMD ["python", "-m", "cog.server.http"]

EXPOSE 5000
