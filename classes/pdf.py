import PyPDF2       # Для чтения PDF
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

    def is_contains_data_table(self) -> bool:
        full_path = f'{self.path_dir}/{self.pdf_file}'

        for pagenum, page in enumerate(extract_pages(pdf_file=full_path)):
            # Проверка элементов на наличие таблиц
            for element in page:
                # Проверка элементов на наличие таблиц
                if isinstance(element, LTRect):
                    return isinstance(element, LTRect)


pdf = PDF(path_dir='../extract_assets', pdf_file='BL24-10003.pdf')
print(pdf.is_contains_data_table())

