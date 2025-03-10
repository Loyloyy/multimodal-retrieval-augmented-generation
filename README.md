# Multi-Modal RAG

This application is an evolution of the text-only RAG. This chatbot incorporates documents and images into a RAG model for comprehensive responses. It offers users the ability to explore the contents of vector stores, providing insights into stored information. Additionally, users have control over vector store creation and deletion, enabling tailored knowledge management.

![alt text](mm1.png "Creating Vector Store")
The above shows a simplified diagram of how a large language model (LLM) can access and use external knowledge. When given a prompt, it doesn't just rely on its internal training data. Instead, it searches through various sources like a knowledge base or the internet to find relevant information. This information is then used to provide a more informed and accurate response.
Essentially, it's like the LLM has a "knowledge assistant" helping it to answer questions and complete tasks. This makes the LLM more powerful and versatile.
- This can be used to ensure LLM returns more updated responses or even sensitive data.
- Of course, the images from documents have to be processed beforehand but that's the secret sauce!


![alt text](mm2.png "RAG in action!")
The demo allows a user to interact with a chatbot that utilizes Retrieval Augmented Generation (RAG) to enhance the performance of a large language model. Conventional RAG only returns text documents but this can return images too!

It also displays options for selecting available LLMs and vector stores, a prompt template for the user to change accordingly. At the bottom, user can also peruse the returned documents and images used to supplement the LLM's output.

![alt text](mm3.png "Semantic image search using a vector database")
A little side project to do image search based on text! I always love explaining this to people.

## 1. Clone this repository

## 2. Create a .env file with the help of the .env.template file

## 3. Build docker image
    
    docker pull vllm/vllm-openai:latest
    docker build -f Dockerfile_frontend -t your-registry/image-name:version .

## 4. Start the vLLM-powered inference API

    docker run --gpus "device=0" --cpus="16" --memory="16g" --shm-size="8gb" \
        --env-file ${pwd}/.env \
        -v ${pwd}:/nfs \
        -p 8001:8001 -it --rm --name app-name \
        vllm/vllm-openai:latest \
        python -m vllm.entrypoints.openai.api_server --host "0.0.0.0" --port "8001" \
        --gpu-memory-utilization 0.5 --model ${pwd}/your-model \

## 5. Start the Gradio frontend
    docker run --cpus="8" --memory="16g" --gpus "device=1" \
        -v ${pwd}:/nfs -p 7860:7860 -it --rm --name app-name \
        --env-file .env \
        your-registry/image-name:version
