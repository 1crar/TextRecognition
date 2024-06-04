import PyPDF2       # Для чтения PDF
# from tabula import read_pdf

from pdfminer.high_level import extract_pages, extract_text     # Для анализа PDF-макета и извлечения текста
from pdfminer.layout import LTTextContainer, LTChar, LTRect, LTFigure   # Для анализа PDF-макета и извлечения текста
import pdfplumber       # Для извлечения текста из таблиц в PDF


class PDF:
    """
    Класс, предназначенный для анализа макета документа
    """
    def __init__(self, path_dir: str, pdf_file: str):
        self.path_dir = path_dir
        self.pdf_file = pdf_file
        self.full_path = f'{self.path_dir}/{self.pdf_file}'

    def is_contains_data_table(self) -> bool:
        for pagenum, page in enumerate(extract_pages(pdf_file=self.full_path)):
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

