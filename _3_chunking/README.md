# Chunking

## Local Running
1. The input is all markdown file stored in folder ```input_md```
2. The output will be stored in folder ```output_json```
3. no need to use any other model, in this case i use regex modification for splitting (for another improvements using model, still in progress)
4. for better running, build the venv in your locals 
5. to run, import all libraries ```pip install -r requirements.txt```
6. Then, run the ```python main.py```

## Fast API running
1. run by the API by activate the venv first, then ```python api.py```
2. i made another file in ```api.py``` for integrating with fastapi without changing the ```main.py```
3. don't forget to open http url in browser and add the ```{url}/docs``` to run the function method available
   
![3_chunking_completed](https://github.com/user-attachments/assets/849e943e-a5fb-46b6-b147-9e745a2f81f5)
