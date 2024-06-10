# Для считывания PDF
import PyPDF2
# Для анализа PDF-макета и извлечения текста
from pdfminer.high_level import extract_pages, extract_text
from pdfminer.layout import LTTextContainer, LTChar, LTRect, LTFigure

# Для извлечения текста из таблиц в PDF (для класса PdfTabula)
import pdfplumber
import tabula

# Для класса PdfCamelot
import pandas as pd
import camelot
# Так импортируется PyMuPDF
import sys, fitz


class PdfAnalyzer:
    """
    Класс, предназначенный для анализа макета документа
    """
    def __init__(self, path_dir: str, pdf_file: str):
        self._path_dir = path_dir
        self._pdf_file = pdf_file
        self._full_path = f'{self._path_dir}/{self._pdf_file}'

    @property
    def full_path(self) -> str:
        return self._full_path

    def is_contains_data_table(self) -> bool:
        for page_num, page in enumerate(extract_pages(pdf_file=self._full_path)):
            # Проверка элементов на наличие таблиц
            for element in page:
                # Проверка элементов на наличие таблиц
                if isinstance(element, LTRect):
                    return isinstance(element, LTRect)
                return False


# Функция для извлечения данных из таблицы pdf. Возвращает список данных
def extract_tables(path_to_pdf: str) -> list:
    pdf_file = pdfplumber.open(path_to_pdf)
    extracted_dt: list = []

    for page_num in range(len(pdf_file.pages)):
        table_page = pdf_file.pages[page_num]
        tables = table_page.extract_tables()

        for table in enumerate(tables):
            # print(f"Table on page {page_num+1}")
            # print(table)
            extracted_dt.append(table)

    return extracted_dt


class PdfTabula:
    """
    Через Tabula не очень :(
    """
    def __init__(self, path_dir: str, pdf_file: str):
        self._path_dir = path_dir
        self._pdf_file = pdf_file
        self._full_path = f'{self._path_dir}/{self._pdf_file}'

    def read_pdf(self) -> list:
        file_path: str = self._full_path
        # Используйте функцию read_pdf для извлечения таблиц из PDF
        tables = tabula.read_pdf(file_path, pages='all', multiple_tables=True)
        # print(tables)
        return tables


class PdfCamelot:
    def __init__(self, path_dir: str, pdf_file: str):
        self._path_dir = path_dir
        self._pdf_file = pdf_file
        self._full_path = f'{self._path_dir}/{self._pdf_file}'

    def convert_to_image(self, png_name: str) -> None:
        """

        :param png_to_path: Даем название файла (сохраняется в 'extract_assets/output_files/')
        :return: None
        """
        folder = 'extract_assets/output_files/'

        # Открываем документ
        doc = fitz.open(self._full_path)
        for page in doc.pages():
            # Переводим страницу в картинку
            pix = page.get_pixmap()
            # Сохраняем
            pix.save(f'{folder}{png_name}')








