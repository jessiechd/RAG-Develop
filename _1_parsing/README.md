# Parsing

## Prepare:
to avoid error in ```package depedency conflict```, make sure :

1. prefer to build venv first by ```python -m venv venv```
2. then install depedencies in ```requirements-torch.txt``` and  ```requirements.txt```
3. only install the preferred depedencies (include the version number for python) - available in ```requirements.txt```
4. recommended to put the input pdf and output md in the same folder
5. running in local environment using ```python main.py```

## Fast API running
1. run by the API by activate the venv first, then ```python api.py```
2. i made another file in ```api.py``` for integrating with fastapi without changing the ```main.py```
3. don't forget to open http url in browser and add the ```{url}/docs``` to run the function method available using swagger ui in the browser


![1_parsing_completed](https://github.com/user-attachments/assets/e3ec57f8-f6d9-4e77-888e-50af20d36138)
