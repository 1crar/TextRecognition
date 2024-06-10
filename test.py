import time
# To read the PDF
import PyPDF2
# To analyze the PDF layout and extract text
from pdfminer.high_level import extract_pages, extract_text
from pdfminer.layout import LTTextContainer, LTChar, LTRect, LTFigure
# To extract text from tables in PDF
import pdfplumber
# To extract the images from the PDFs
from PIL import Image
from pdf2image import convert_from_path
# To perform OCR to extract text from images
import pytesseract
import json

from dotenv import load_dotenv
import os
# Загружаем переменные из файла .env
load_dotenv()

path_to_tesseract: str = os.getenv('TESSERACT_PATH_DIR')
tesseract_exe: str = os.getenv('EXECUTION_FILE')
# If you don't have plopper executable in your PATH, include the following:
poppler_path: str = os.getenv('POPPLER_PATH')
# If you don't have tesseract executable in your PATH, include the following:
pytesseract.pytesseract.tesseract_cmd: str = f'{path_to_tesseract}/{tesseract_exe}'


def text_extraction(element: "pdfminer.layout.LTTextLineHorizontal") -> tuple[str, list[float]]:
    """
    Работает с извлеченным текстом, который ранее определился как сущность LTTextContainer, т.е. исключительно текст
    который был изъят НЕ из картинок и таблиц.
    """
    # Извлекаем текст из вложенного текстового элемента
    line_text: str = element.get_text()

    # Находим форматы текста
    # Инициализируем список со всеми форматами, встречающимися в строке текста
    line_formats: list = []

    for text_line in element:
        if isinstance(text_line, LTTextContainer):
            # Итеративно обходим каждый символ в строке текста
            for character in text_line:
                if isinstance(character, LTChar):
                    # Добавляем к символу название шрифта
                    line_formats.append(character.fontname)
                    # Добавляем к символу размер шрифта
                    line_formats.append(character.size)
    # Находим уникальные размеры и названия шрифтов в строке
    format_per_line = list(set(line_formats))

    # Возвращаем кортеж с текстом в каждой строке вместе с его форматом
    return (line_text, format_per_line)


def extract_table(pdf_path: str, page_num: int, table_num: int) -> list[list[str | None]]:
    """
    Мы открываем файл PDF.
    Переходим к исследуемой странице файла PDF.
    Из списка таблиц, найденных на странице библиотекой pdfplumber, мы выбираем нужную нам.
    Извлекаем содержимое таблицы и выводим его в список вложенных списков, представляющих собой каждую строку таблицы.
    """
    # Открываем pdf файл
    pdf = pdfplumber.open(pdf_path)
    # Находим исследуемую страницу
    table_page = pdf.pages[page_num]
    # Извлекаем соответствующую таблицу
    table: list = table_page.extract_tables()[table_num]
    return table


# Convert table into appropriate format
def table_converter(table: list) -> str:
    """
    Мы итеративно обходим каждый вложенный список и очищаем его содержимое от ненужных разрывов строк, возникающих
    из-за текста с переносами.
    Объединяем каждый элемент строки таблицы, разделяя их символом | для создания структуры ячейки таблицы.
    Наконец, мы добавляем разрыв строки в конце, чтобы перейти к следующей строке таблицы.
    """
    table_string = ''
    # Итерируемся по каждой строке в таблицу
    for row_num in range(len(table)):
        row = table[row_num]
        # Удаляем разрыв строки из текста с переносом
        cleaned_row = [
            item.replace('\n', ' ') if item is not None and '\n' in item else 'None' if item is None else item for item
            in row]
        # Преобразуем таблицу в строку
        table_string += ('|' + '|'.join(cleaned_row) + '|' + '\n')
    # Удаляем последний разрыв строки
    table_string = table_string[:-1]
    return table_string


# Функция для проверки наличия таблиц в текщей странице pdf
def is_any_table(element: "pdfminer.layout.LTTextLineHorizontal", page: int, tables: list) -> bool:
    x0, y0up, x1, y1up = element.bbox
    # Меняем координаты, так как pdfminer считывает страницу снизу вверх
    y0 = page.bbox[3] - y1up
    y1 = page.bbox[3] - y0up
    for table in tables:
        tx0, ty0, tx1, ty1 = table.bbox
        if tx0 <= x0 <= x1 <= tx1 and ty0 <= y0 <= y1 <= ty1:
            return True
    return False


# Функция нахождения таблицы текущего элемента (объекта "pdfminer.layout.LTTextLineHorizontal")
def find_table_for_element(element: "pdfminer.layout.LTTextLineHorizontal", page: int, tables: list) -> int | None:
    x0, y0up, x1, y1up = element.bbox
    # Меняем координаты, так как pdfminer считывает каждую страницу снизу вверх
    y0 = page.bbox[3] - y1up
    y1 = page.bbox[3] - y0up
    for i, table in enumerate(tables):
        tx0, ty0, tx1, ty1 = table.bbox
        if tx0 <= x0 <= x1 <= tx1 and ty0 <= y0 <= y1 <= ty1:
            # Возвращаем индекс таблицы
            return i
    # Или None, если таблицы нет
    return None


# Создаём функцию для вырезания элементов изображений из PDF
def crop_image(element: "pdfminer.layout.LTTextLineHorizontal", pageObj: dict) -> None:
    """
    Функция, предназначенная для обрезания элемента - картинки, для дальнейшей ее обработки
    """
    # Получаем координаты для вырезания изображения из PDF
    [image_left, image_top, image_right, image_bottom] = [element.x0,element.y0,element.x1,element.y1]
    # Обрезаем страницу по координатам (left, bottom, right, top)
    pageObj.mediabox.lower_left = (image_left, image_bottom)
    pageObj.mediabox.upper_right = (image_right, image_top)
    # Сохраняем обрезанную страницу в новый PDF
    cropped_pdf_writer = PyPDF2.PdfWriter()
    cropped_pdf_writer.add_page(pageObj)
    # Сохраняем обрезанный PDF в новый файл
    with open('extract_assets/cropped_outputs/cropped_image.pdf', 'wb') as cropped_pdf_file:
        cropped_pdf_writer.write(cropped_pdf_file)


# Создаём функцию для преобразования PDF в изображения
def convert_to_images(input_file: str,):
    images = convert_from_path(input_file, poppler_path=poppler_path)
    image = images[0]
    output_file = 'extract_assets/output_files/PDF_image.png'
    image.save(output_file, 'PNG')


# Создаём функцию для считывания текста из изображений
def image_to_text(image_path: str) -> str:
    # Считываем изображение
    img = Image.open(image_path)
    # Извлекаем текст из изображения
    text = pytesseract.image_to_string(img, lang='rus')
    return text


start = time.time()
# Устанавливаем путь к PDF
pdf_path = 'extract_assets/input_files/УПД.pdf'

# создаём объект файла PDF
pdfFileObj: "io.BufferedReader" = open(pdf_path, 'rb')
# создаём объект считывателя PDF
pdfReaded: "PyPDF2._reader.PdfReader" = PyPDF2.PdfReader(pdfFileObj)

# Создаём словарь для извлечения текста из каждого изображения
text_per_page: dict = {}
# Извлекаем страницы из PDF
for pagenum, page in enumerate(extract_pages(pdf_path)):

    # Инициализируем переменные, необходимые для извлечения текста со страницы
    pageObj: dict = pdfReaded.pages[pagenum]
    page_text: list = []
    line_format: list = []
    text_from_images: list = []
    text_from_tables: list = []
    page_content: list = []
    # Инициализируем количество исследованных таблиц
    table_num: int = 0
    first_element: bool = True
    table_extraction_flag: bool = False
    # Открываем файл pdf
    pdf: "pdfplumber.pdf.PDF" = pdfplumber.open(pdf_path)
    # Находим исследуемую страницу
    page_tables: "pdfplumber.page.Page" = pdf.pages[pagenum]
    # Находим количество таблиц на странице
    tables: list["pdfplumber.table.Table"] = page_tables.find_tables()

    # Находим все элементы страницы pdf
    page_elements: list[tuple] = [(element.y1, element) for element in page._objs]
    # Сортируем все элементы по порядку нахождения на странице
    page_elements.sort(key=lambda a: a[0], reverse=True)

    # Находим элементы, составляющие страницу
    for i, component in enumerate(page_elements):
        # Извлекаем положение верхнего края элемента в PDF, pos - позиция текущего итерируемого элемента
        pos = component[0]
        # Извлекаем элемент структуры страницы
        element = component[1]

        # Проверяем, является ли элемент текстовым
        if isinstance(element, LTTextContainer):
            # Проверяем, находится ли текст в таблице
            if not table_extraction_flag:
                # Используем функцию извлечения текста и формата для каждого текстового элемента
                (line_text, format_per_line) = text_extraction(element)
                # Добавляем текст каждой строки к тексту страницы
                page_text.append(line_text)
                # Добавляем формат каждой строки, содержащей текст
                line_format.append(format_per_line)
                page_content.append(line_text)
            else:
                # Пропускаем текст, находящийся в таблице
                pass

        # Проверяем элементы на наличие изображений
        if isinstance(element, LTFigure):
            # Вырезаем изображение из PDF
            crop_image(element, pageObj)
            # Преобразуем обрезанный pdf в изображение
            convert_to_images('extract_assets/cropped_outputs/cropped_image.pdf')
            # Извлекаем текст из изображения
            image_text = image_to_text('extract_assets/output_files/PDF_image.png')
            text_from_images.append(image_text)
            page_content.append(image_text)
            # Добавляем условное обозначение в списки текста и формата
            page_text.append('image')
            line_format.append('image')

        # Проверяем элементы на наличие таблиц
        if isinstance(element, LTRect):
            # Если первый прямоугольный элемент
            if first_element and (table_num + 1) <= len(tables):
                # Находим ограничивающий прямоугольник таблицы
                lower_side = page.bbox[3] - tables[table_num].bbox[3]
                upper_side = element.y1
                # Извлекаем информацию из таблицы
                table = extract_table(pdf_path, pagenum, table_num)
                # Преобразуем информацию таблицы в формат структурированной строки
                table_string = table_converter(table)
                # Добавляем строку таблицы в список
                text_from_tables.append(table_string)
                page_content.append(table_string)
                # Устанавливаем флаг True, чтобы избежать повторения содержимого
                table_extraction_flag = True
                # Преобразуем в другой элемент
                first_element = False
                # Добавляем условное обозначение в списки текста и формата
                page_text.append('table')
                line_format.append('table')

            # Проверяем, извлекли ли мы уже таблицы из этой страницы
            try:
                if element.y0 >= lower_side and element.y1 <= upper_side:
                    pass
                elif not isinstance(page_elements[i + 1][1], LTRect):

                    table_extraction_flag = False
                    first_element = True
                    table_num += 1
            except IndexError as e:
                print(f'Detected error {e}')

    # Создаём ключ для словаря
    dctkey = 'Page_' + str(pagenum)
    # Добавляем список списков как значение ключа страницы
    text_per_page[dctkey] = [page_text, line_format, text_from_images, text_from_tables, page_content]

# Закрываем объект файла pdf
pdfFileObj.close()

print(text_per_page, text_per_page['Page_0'], sep='\n')

with open('extracted_results/PyPdf_result.txt', mode='w', encoding='utf-8') as file:
    # Дублирование извлеченного текста (Почему)
    list_from_text_per_page: list = text_per_page['Page_0']

    for internal_list in list_from_text_per_page:
        for el in internal_list:
            file.writelines(str(el))

print('---------------Execution time---------------', (time.time() - start), sep='\n')
