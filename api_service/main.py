from fastapi import FastAPI

app = FastAPI()


@app.get('/')
async def get_page():
    return {"Hello": "world"}
