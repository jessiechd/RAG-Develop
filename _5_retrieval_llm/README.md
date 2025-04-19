# Retrieval and Chat with LLM

## Local Running
1. install the ```requirements.txt``` first in your environment
2. fill the ```.env``` with supabase url and the supabase API key and supabase DB connection, also the openai API key
3. for local running, run this in the terminal ```python main.py```, it will take longer time because nltk downloads, but for the second running it will be normal
4. when it runs, user can input the queries, and still continuing the conversation with the chat_history.
5. If user give the query "new chat" it will delete all chat_history and start a new conversation
6. If user give the query "exit" it will close the terminate the conversation-session
   ![5_terminal](https://github.com/user-attachments/assets/46f75a63-8aab-4946-8b55-96a4dee30a9b)


## Fast API Running
1. run by the API by activate the venv first, then ```python api.py```
2. i made another file in ```api.py``` for integrating with fastapi without changing the ```main.py```
3. don't forget to open http url in browser and add the ```{url}/docs``` to run the function method available
4. there are two endpoints available, the ```\query``` for taking top-5 most related chunks. Then, the ```\chat``` endpoint to take the answer from LLM
   ![screencapture-127-0-0-1-8400-docs-2025-04-15-20_22_07](https://github.com/user-attachments/assets/c57ca0cf-5101-4d61-a3d1-bdf569f90467)

   
