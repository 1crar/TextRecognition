import io
import PyPDF2

from fastapi import APIRouter, UploadFile

from app.pdf_appRecognizer.pdf_app import pdf_app

pdf_service = APIRouter()


@pdf_service.get("/")
async def router_main():
    return {"router": "pdfService_route"}


@pdf_service.post("/upload_pdf/", status_code=201)
async def upload_file(uploaded_file: UploadFile):
    # В переменной content содержится байт-код содержимого pdf (полученного через POST запросы)
    content = await uploaded_file.read()
    # Записываем файл (сохраняем содержимое файла)
    pdf_reader = PyPDF2.PdfFileReader(io.BytesIO(content))
    pdf_writer = PyPDF2.PdfFileWriter()
    # Проходимся по каждой странице и добавляем ее к pdf_writer
    for page_num in range(pdf_reader.getNumPages()):
        pdf_writer.add_page(pdf_reader.getPage(page_num))
    # Записываем полученное имя файла
    file_name: str = uploaded_file.filename
    with open(f'uploaded_files/{file_name}', 'wb') as output_pdf:
        pdf_writer.write(output_pdf)
    # Вызываем приложение pdf recognizer
    data: dict = pdf_app(pdf_filename=file_name)
    return {"pdf_filename": file_name,
            "data": data}
