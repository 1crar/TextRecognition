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


# Создаем экземпляр класса
pdf = PDF(path_dir='../extract_assets', pdf_file='BL24-union.pdf')
# Извлечение данных табличной части
if pdf.is_contains_data_table():
    pdf_path = '../extract_assets/BL24-union.pdf'


    def extract_tables(pdf_path: str) -> list:
        pdf = pdfplumber.open(pdf_path)
        extracted_table: list = []

        for page_num in range(len(pdf.pages)):
            table_page = pdf.pages[page_num]
            tables = table_page.extract_tables()

            for table in enumerate(tables):
                # print(f"Table on page {page_num+1}")
                # print(table)
                extracted_table.append(table)

        return extracted_table

    pdf_table = extract_tables(pdf_path)

    dt = []             # список двумерных массивов (двумерных списков)
    for el in pdf_table:
        if type(el) == list:
            dt.append(el)

    for i in range(len(pdf_table)):
        dt.append(pdf_table[i][1])

    print(pdf_table[0][1], dt, sep='\n')
else:
    print("Таблица не найдена.")

