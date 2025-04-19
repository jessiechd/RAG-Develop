# n8n Integrations

## Build the Knowledge Base
1. include the process for ```PARSING```, ```IMAGE DESCRIPTION```, ```CHUNKING```, ```STORE EMBEDDING IN SUPABASE```
2. to import the workflow, import the ```knowledge_base.json``` file into new workflow in n8n
3. integrate with customable API url for each stage in RAG stated in point 1


## Chat Interface 
1. build the chat interface from last stage in RAG in ```RETRIEVAL``` and ```Chat LLM```
2. to import the workflow, import the ```chatbot.json``` file into new workflow in n8n
3. integrate with customable API url for last stage in RAG in 5_retrieval_llm

### Knowledge base Workflow
![n8n_onebyone](https://github.com/user-attachments/assets/9e5972a7-0628-46e1-9225-e05b0f376cb5)
![n8n_parsing](https://github.com/user-attachments/assets/6cfc0fb1-3b43-467a-90cc-ed1668b15751)
![n8n_image](https://github.com/user-attachments/assets/ad3d4e25-0737-4b25-94d1-b139d670707b)
![n8n_chunking](https://github.com/user-attachments/assets/0c72e3f6-a466-46e8-b788-509632810b95)
![n8n_embedding-store](https://github.com/user-attachments/assets/3cb097fa-9f87-4d7a-bb46-e45cea55ab70)

### Chat Interface
![n8n--1](https://github.com/user-attachments/assets/f559df22-5a4c-47f1-87c7-aefb2cf136c7)
