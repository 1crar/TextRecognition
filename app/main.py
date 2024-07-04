from fastapi import FastAPI

from routers.pdf_service import pdf_service

app = FastAPI()
app.include_router(router=pdf_service, prefix="/pdf_service", tags=["Api app"])


@app.get('/')
async def get_page():
    return {"main": "page"}
