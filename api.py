import os
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import importlib.util
import requests
import shutil
import os

def import_router(path):
    spec = importlib.util.spec_from_file_location("api", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.router

parsing_router = import_router(os.path.abspath("_1_parsing/api.py"))
image_router = import_router(os.path.abspath("_2_image/api.py"))



# parsing_app = import_api(os.path.abspath("_1_parsing/api.py"))
# image_app = import_api(os.path.abspath("_2_image/api.py"))
# chunking_app = import_api(os.path.abspath("_3_chunking/api.py"))
# embedding_app = import_api(os.path.abspath("_4_embedding_store/api.py"))
# llm_app = import_api(os.path.abspath("_5_retrieval_llm/api.py"))

app = FastAPI(title="Pipeline API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(parsing_router, prefix="/upload")
app.include_router(image_router, prefix="/img")



# app.mount("/upload", parsing_app)
# app.mount("/img", image_app)
# app.mount("/markdown", chunking_app)
# app.mount("/store", embedding_app)
# app.mount("/llm", llm_app)


@app.get("/")
def read_root():
    return {"message": "Pipeline API is running!"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9000)





# @app.post("/upload_full")
# async def upload_full(file: UploadFile = File(...)):

#     temp_file_path = f"./_1_parsing/input_pdfs/{file.filename}"
#     with open(temp_file_path, "wb") as buffer:
#         shutil.copyfileobj(file.file, buffer)


#     res_parsing = requests.post("http://localhost:9000/parsing/parsing", files={"file": open(temp_file_path, "rb")})
#     parsing_result = res_parsing.json()
#     md_filename = parsing_result["markdown_file"]
#     md_path = f"./_1_parsing/output_md/{md_filename}"


#     res_image = requests.post("http://localhost:9000/image_description", json={"path": md_path})
#     updated_md_path = res_image.json()["output_markdown"]


#     res_chunking = requests.post("http://localhost:9000/chunking", json={"path": updated_md_path})
#     json_path = res_chunking.json()["json_output"]


#     res_embedding = requests.post("http://localhost:9000/embedding", json={"path": json_path})

#     return {
#         "status": "done",
#         "markdown": updated_md_path,
#         "chunks": json_path,
#         "embedding_result": res_embedding.json()
#     }

