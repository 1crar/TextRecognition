import csv
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


class PdfTabula(PdfAnalyzer):
    """
    Через Tabula не очень :(
    """

    # Наследуем метод __init__ из PdfAnalyzer
    def __init__(self, path_dir: str, pdf_file: str):
        super().__init__(path_dir, pdf_file)

    def read_pdf(self) -> list:
        file_path: str = self._full_path
        # Используйте функцию read_pdf для извлечения таблиц из PDF
        tables = tabula.read_pdf(file_path, pages='all', multiple_tables=True)
        # print(tables)
        return tables


class PdfCamelot(PdfAnalyzer):
    """
    Класс, предназначенный для извлечения табличной части пдф и записи извлеченных данных в csv файл
    """

    # Наследуем метод __init__ из PdfAnalyzer
    def __init__(self, path_dir: str, pdf_file: str):
        super().__init__(path_dir, pdf_file)

    def read_tables(self, num_page: str = 'all') -> "camelot.core.TableList":
        """

        :param num_page: Необязательный параметр, all по умолчанию
        :return: возвращает объект camelot.core.TableList (по сути, список таблиц)
        """
        tables = camelot.read_pdf(
            filepath=self._full_path,
            pages=num_page)
        return tables

    @staticmethod
    def write_to_csv(tables: "camelot.core.TableList", file_csv_name: str) -> None:
        folder = 'extracted_results/'

        with open(file=f'{folder}{file_csv_name}', mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)

            for table in tables:

                cleaned_table = []
                for row in table.data[1:]:
                    cleaned_row = [item.replace('\n', ' ') for item in row]
                    cleaned_table.append(cleaned_row)
                print(cleaned_table)

                for row in cleaned_table:
                    writer.writerow(row)


class PdfTextReader(PdfAnalyzer):
    """
    Предназначен только для чтения текста из pdf файла:
    """

    # Наследуем метод __init__ из PdfAnalyzer
    def __init__(self, path_dir: str, pdf_file: str):
        super().__init__(path_dir, pdf_file)

    def extract_text_from_pdf(self) -> str:
        text = ''
        with open(self._full_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfFileReader(file)
            num_pages = pdf_reader.numPages

            for page_num in range(num_pages):
                page = pdf_reader.getPage(page_num)
                text += page.extract_text()
        return text













