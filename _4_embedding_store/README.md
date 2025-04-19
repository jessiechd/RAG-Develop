# Store Embedding in Supabase

## Supabase preparation:
### Schema : public
1. build a project
2. make a table ```documents```, checklist the ```Enable Row Level Security``` and set with a column :
```
id : int8; -> primary key
content : text;
metadata : jsonb;
type : text;
chunk_id : uuid
```
![supabase_documents](https://github.com/user-attachments/assets/938c30ff-ae2a-4b42-9d67-c45d023176d2)

3. disable the RLS (Row Level Security) by clicking ```Add RLS policy -> 
Disable RLS -> Confirm```
4. make a table ```tables```, cheklist the ```Enable Row Level Security``` and set with a column :
```
chunk_id : uuid -> primary key
table_data : jsonb;
metadata : jsonb;
description : text
```
7. disable the RLS (Row Level Security) by clicking ```Add RLS policy -> 
Disable RLS -> Confirm```
![supabase_documents](https://github.com/user-attachments/assets/ebeb4b3e-39c1-4f92-9885-50dd4da86a85)

### Schema : vecs
1. You dont need to manually make a new schema, it already build by the code in the main.py
2. The vecs schema will store the vector as the result of embedding
3. Overview :
   ![supabase_vecs](https://github.com/user-attachments/assets/b386d26a-56c5-40d4-88a5-498baff4629e)


# Local Running
1. install the ```requirements.txt``` first in your environment
2. fill the ```.env``` supabase url and the supabase API key and supabase DB connection here
3. make sure the input folder in the same folder as code and other dependencies
4. If the venv supports GPU, make sure install the PyTorch with the correct CUDA version. (my system supports CUDA 12.5, so i run this ```pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121```)
5. if find some error while loading the model, make sure to test your env supported GPU or not, by running the ```test.py``` first. Expected result :
   ```
   "CUDA Available: True"
   "GPU Name: NVIDIA GeForce RTX 3050".
   ```
   if you have GPU but its not detected, run ```nvidia-smi``` in the cmd first
7. the code also can be run in the CPU
8. for local running, run this in the terminal ```python main.py```

## Fast API running
1. run by the API by activate the venv first, then ```python api.py```
2. i made another file in ```api.py``` for integrating with fastapi without changing the ```main.py```
3. don't forget to open http url in browser and add the ```{url}/docs``` to run the function method available
   ![screencapture-127-0-0-1-8300-docs-2025-04-15-20_24_36](https://github.com/user-attachments/assets/80851809-2ab4-4270-979e-5ec1a69969dd)
