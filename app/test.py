import cv2
import imutils
import json
import os

import pandas as pd
import pytesseract

import numpy as np

from dotenv import load_dotenv
from img2table.document import Image
from pytesseract import Output
from PIL import Image, ImageEnhance

from pdf_appRecognizer.classes.converter import Converter
from pdf_appRecognizer.classes.pdf import PdfCamelot, PdfTextReader
from pdf_appRecognizer.classes.data_json import DataCleaning, DataCollection, DictToJson
from pdf_appRecognizer.classes.text_extracter import DataExtractionImage, DataExtractionPDF
from pdf_appRecognizer.classes.img import ImageDataExtracter


load_dotenv()
TESSERACT_OCR: str = os.getenv('TESSERACT')


def improve_img_quality(img_path: str, output_path: str, sharpness: int = 10, contrast: float = 0.9, blur: int = 1):
    """
    Функция улучшения качества изображения. Ненамного, ну качество улучшает
    :param img_path: Путь до картинки, которую собираемся улучшать
    :param output_path: Путь для сохранения обработанной картинки
    :param sharpness: Степень резкости
    :param contrast: Степень контрастности
    :param blur: Степень размытия
    :return: None

    PS. Можно попробовать поиграться с параметрами для sharpness, contrast и blur: int = 1.
    """

    # Загружаем изображение
    img = cv2.imread(img_path)

    # Придаем серый оттеннок
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # конвертируем image в PIL Image
    pil_img = Image.fromarray(img)

    # Увеличиваем резкость изображения
    enhancer = ImageEnhance.Sharpness(pil_img)
    img_enhanced = enhancer.enhance(sharpness)

    # Увеличиваем контрастность
    enhancer = ImageEnhance.Contrast(img_enhanced)
    img_enhanced = enhancer.enhance(contrast)

    # Конвертируем в OpenCV image (numpy массив - array)
    img_enhanced = np.array(img_enhanced)

    # Применяем небольшое размытие
    img_enhanced = cv2.GaussianBlur(img_enhanced, (blur, blur), 0)

    # Конвертируем в PIL Image и сохраняем
    img_enhanced = Image.fromarray(img_enhanced)
    img_enhanced.save(output_path)


def prep_image_var():
    """
    Эта функция выделяет именно табличную часть. Но тестировал тольк для одностраничных УПД
    :return: None
    """
    img = cv2.imread(r'pdf_appRecognizer/extract_assets/image_files/Test_YPD_2.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    # Выделяем контуры для табличной части
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Перебор найденных контуров
    for cnt in contours:
        area = cv2.contourArea(cnt)

        # Фильтрация по площади контура
        if area > 1000:  # Пороговое значение для фильтрации
            # Создание маски для выделения контура
            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.drawContours(mask, [cnt], -1, (255), thickness=cv2.FILLED)

            # Извлечение области таблицы
            table_image = cv2.bitwise_and(img, img, mask=mask)
            cv2.imwrite('pdf_appRecognizer/extract_assets/image_files/Test_Table_YPD_2.png', img=table_image)

            table_image_2 = cv2.imread('pdf_appRecognizer/extract_assets/image_files/Test_Table_YPD_2.png')
            gray_img = cv2.cvtColor(table_image_2, cv2.COLOR_BGR2GRAY)
            cv2.imwrite('pdf_appRecognizer/extract_assets/image_files/Test_Table_YPD_2_2.png', img=gray_img)


def data_write_collection(collection: dict, path: str) -> None:
    # функция для записи извлеченных значений из изображения в формате .json (НЕ табличная часть)
    """

    :param collection: Хэш-таблица/словарь
    :param path: Путь сохранения json файла
    :return:
    """
    try:
        with open(file=f'{path}/extracted_data.json', mode='w', encoding='utf-8') as json_file:
            json.dump(obj=collection, fp=json_file, ensure_ascii=False, indent=3)
    except Exception as e:
        raise Exception(e)


def test_pdf_to_image():
    """
    Функция для конвертирования из пдф в пнг и обратно. Не нужна, если брать пнг УПД из инета.
    :return:
    """
    Converter.from_pdf_to_png(pdf_path='pdf_appRecognizer/extract_assets/pdf_files/УПД.pdf',
                              to_save='pdf_appRecognizer/extract_assets/image_files/УПД')

    Converter.from_png_to_pdf(png_path='pdf_appRecognizer/extract_assets/image_files/УПД_1.png',
                              to_save='pdf_appRecognizer/extract_assets/pdf_files/УПД_1.pdf')


def test_converted_pdf_for_extracting():
    """
    А тут я тестировал конвертированный пдф (пнг -> пдф) на извлечение. К сожалению, камелотом вообще ничего не извле-
    кает.
    :return: None
    """

    # Ф-ия для конвертирования из pdf в png. Сейчас не используется, так как примеры упд в формате пнг я беру из инета
    camelot_instance = PdfCamelot(path_dir='pdf_appRecognizer/extract_assets/pdf_files',
                                  pdf_file='УПД_1.pdf')
    # Считываем таблицы с помощью камелота
    tables = camelot_instance.read_tables()

    # Получаем кол-во таблиц для обработки
    table_numbers: int = tables.n
    if table_numbers == 1:
        table: list = tables[0].data
    else:
        table: list = []
        for i in range(0, table_numbers):
            table += tables[i].data

    # Затем импортируем класс DataCleaning для очистки и работы с извлеченной таблицей
    cleaned_table: list = DataCleaning.data_clean(data_table=table)

    # Далее извлекаем текст из pdf (без учета структуры) для извлечения данных вне табличной части
    # (ИНН/КПП, Счет-фактура)
    pdf = PdfTextReader(path_dir='pdf_appRecognizer/extract_assets/pdf_files',
                        pdf_file='УПД.pdf')

    # Создаем экземпляр класса текст
    text = pdf.extract_text_from_pdf()
    # Извлекаем текст из pdf
    my_regulars: 'DataExtraction' = DataExtractionPDF(text=text)

    # Извлекаем остальные данные (ИНН/КПП, Счет-фактура)
    inn_kpp: str = my_regulars.inn_and_kpp_extract()
    invoice: str = my_regulars.invoice_extract()

    # Извлекаем значение всего к оплате
    totals: tuple = my_regulars.total_sum_extract(data_table=cleaned_table)
    # Формируем хэш-таблицу на основе полученных данных
    data = DataCollection()
    collection = data.data_collect(inn_kpp=inn_kpp, invoice=invoice, cleaned_data=cleaned_table, totals=totals)

    # Записываем словарь в json-файл
    DictToJson.write_to_json(collection=collection, path_to_save='pdf_appRecognizer/extract_assets/json_files')


def image_extracting(image_file: str):
    """
    Эта функция через регулярки извлекает данные ВНЕ таблицы. Работает по НЕСТРУКТУРИРОВАННОМУ тексту
    :param image_file: Путь до файла
    :return: None
    """

    # Создаем экземпляр класса curr_image для работы с ним. Структура таблицы в таком варианте теряется.
    curr_image = ImageDataExtracter(path_dir='pdf_appRecognizer/extract_assets/image_files',
                                    image_file=image_file,
                                    path_to_tesseract=TESSERACT_OCR, language='rus')

    extracted_text = curr_image.tesseract_extraction()
    # Вывод извлеченного текста (без структуры)
    print(extracted_text, '\n\n')
    # return

    # Далее создаем экземпляр класса text_instance для дальнейшего извлечения полей через регулярки.
    text_instance = DataExtractionImage(text=extracted_text)

    # Далее с помощью класса DataExtractionImage извлекаю данные. файл с классом находится в app/pdf_appRecognizer/classes/text_extracter.py
    inn_kpp: str = text_instance.inn_and_kpp_extract()
    seller: str = text_instance.seller_extract()
    invoice: str = text_instance.invoice_extract()
    adress: str = text_instance.adress_extract()
    loaded_date: str = text_instance.loaded_document_extract()

    data_collection: dict = {
        'Счет-фактура_№': invoice,
        'Продавец': seller,
        'ИНН/КПП': inn_kpp,
        'Адрес': adress,
        'Документ_об_отгрузке': loaded_date
    }
    # Записываем в файл json
    data_write_collection(collection=data_collection, path='pdf_appRecognizer/extract_assets/json_files')

    print(f'ИНН/КПП продавца: {inn_kpp}', f'Продавец: {seller}', f'Счет-фактура № {invoice}',
          f'Адресс: {adress}', f'Документ об отгрузке: {loaded_date}', sep='\n')


"""
А далее уже я тестировал функции на качество извлечения и улучшения изображений.
"""


# prep_image_var()

# improve_img_quality(img_path='pdf_appRecognizer/extract_assets/image_files/Test_Table_YPD_2.png',
#                     output_path='pdf_appRecognizer/extract_assets/image_files/enhanced_Test_Table_YPD_2.png')
image_extracting(image_file='enhanced_Test_Table_YPD_2.png')

# table2img(path_to_img='pdf_appRecognizer/extract_assets/image_files/enhanced_Test_Table_YPD_2.png')

