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

    def is_contains_data_table(self) :
        full_path = f'{self.path_dir}/{self.pdf_file}'

        for pagenum, page in enumerate(extract_pages(pdf_file=full_path)):
            # Проверка элементов на наличие таблиц
            for element in page:
                # Проверка элементов на наличие таблиц
                if isinstance(element, LTRect):
                    return isinstance(element, LTRect)


pdf = PDF(path_dir='../extract_assets', pdf_file='BL24-10003.pdf')

if pdf.is_contains_data_table():
    pdf_path = '../extract_assets/BL24-10003.pdf'

    def extract_table(pdf_path, page_num, table_num):
        pdf = pdfplumber.open(pdf_path)
        # Найти исследуемую страницу
        table_page = pdf.pages[page_num]
        # Извлечение соответствующей таблицы
        table = table_page.extract_tables()[table_num]
        print(table, type(table), sep='\n')

    extract_table(pdf_path, page_num=0, table_num=0)

else:
    print("Таблица не найдена.")
