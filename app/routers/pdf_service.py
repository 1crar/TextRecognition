import io
import PyPDF2

from fastapi import APIRouter, UploadFile


pdf_service = APIRouter()


@pdf_service.get("/")
async def router_main():
    return {"router": "pdfService_route"}
