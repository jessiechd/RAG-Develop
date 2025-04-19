# Image Description

## Prepare:
To run in local:
1. install the ```requirements.txt``` first in your environment
2. make sure the input and output folder in the same folder as code and other depedencies
3. If the env supports GPU, make sure install the **PyTorch with the correct CUDA version**. (my sistem supports system supports CUDA 12.5, so i run this ```pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121```)
4. if find some error while loading the model, make sure to test your env supported GPU or not, by running the "test.py" first.
   Expected result :
```
  "CUDA Available: True"
  "GPU Name: NVIDIA GeForce RTX 3050"
```
   if you have GPU but its not detected, run ```nvidia-smi```in the cmd first to check
   
6. the code also can be run in the CPU, but it may takes longer time to process
7. some arguments errors were solved this time, but maybe it can show problems in another env, so make sure all of them are run in the same env

## Fast API running
1. run by the API by activate the venv first, then ```python api.py```
2. i made another file in ```api.py``` for integrating with fastapi without changing the ```main.py```
3. don't forget to open http url in browser and add the ```{url}/docs``` to run the function method available

![2_image_completed](https://github.com/user-attachments/assets/241bf45b-32b9-45ff-b1de-6b1e94c60508)
