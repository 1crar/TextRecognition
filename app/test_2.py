import os
import cv2
import time

from dotenv import load_dotenv
# from cv2.dnn_superres import DnnSuperResImpl
from img2table.ocr import TesseractOCR
from img2table.document import Image
from PIL import Image as PILImage

load_dotenv()

# src = "pdf_appRecognizer/extract_assets/image_files/Test_Table_YPD_2.png"
#
# # Instantiation of OCR
# #pytesseract.pytesseract.tesseract_cmd = r'C:\Users\melada\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'
# ocr = TesseractOCR(lang="rus")
#
# # Instantiation of document, either an image or a PDF
# doc = Image(src)
# extracted_tables = doc.extract_tables(ocr=ocr)
#
# table_img = cv2.imread(src)
#
# for table in extracted_tables:
#     for row in table.content.values():
#         for cell in row:
#             cv2.rectangle(table_img, (cell.bbox.x1, cell.bbox.y1), (cell.bbox.x2, cell.bbox.y2), (255, 0, 0), 2)
#
# print(PILImage.fromarray(table_img).save("pdf_appRecognizer/extract_assets/image_files/temp_2.png"))
# # Table extraction
# extracted_tables = doc.extract_tables(ocr=ocr,
#                                      implicit_rows=True,
#                                      borderless_tables=False,
#                                      min_confidence=50)
#
# doc.to_xlsx(dest='pdf_appRecognizer/extract_assets/xlsx_files/test.xlsx',
#             ocr=ocr)


def improve_quality(path_to_model: str, path_to_image: str, path_to_save: str) -> None:
    """
    А вот эта функция тоже предназначена для улучшения качества изображения. Используется модель EDSR_x3.pb/EDSR_x4.pb
    Модели были взяты тут: https://github.com/Saafke/EDSR_Tensorflow/tree/master/models

    :param path_to_model: Путь до модели
    :param path_to_image: Путь до изображения
    :param path_to_save: Путь для сохранения обработанного изображения через обученную модель.
    :return: None
    """

    img = cv2.imread(path_to_image)
    sr = cv2.dnn_superres.DnnSuperResImpl()
    sr.readModel(path_to_model)

    sr.setModel("edsr", 4)

    result = sr.upsample(img)
    cv2.imwrite(img=result, filename=path_to_save)


start_time = time.time()
improve_quality(path_to_model='pdf_appRecognizer/extract_assets/models/EDSR_x4.pb',
                path_to_image='pdf_appRecognizer/extract_assets/image_files/Test_YPD_2.png',
                path_to_save='pdf_appRecognizer/extract_assets/image_files/temp_2.png')
end_time = time.time() - start_time
print(f"время выполнение программы: {end_time} сек.")
