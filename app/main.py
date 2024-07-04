import uvicorn

from fastapi import FastAPI

from routers.pdf_service import pdf_service
from app.configs.settings import settings

app = FastAPI()
app.include_router(router=pdf_service, prefix="/pdf_service", tags=["Api app"])


@app.get('/')
async def get_page():
    return {"main": "page"}


if __name__ == '__main__':
    uvicorn.run(
        app="app.main:app",
        reload=settings.RELOAD,
        host=settings.HOST,
        port=settings.PORT,
        log_level="debug"
    )
