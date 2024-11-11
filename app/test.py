import cv2
import imutils
import json
import os
import torch

import pandas as pd
import pytesseract
import numpy as np
import matplotlib.pyplot as plt

from dotenv import load_dotenv
from img2table.document import Image
from img2table.ocr import TesseractOCR
from pytesseract import Output
from PIL import ImageEnhance, ImageFilter
from PIL import Image as Img
from super_image import EdsrModel, DrlnModel, ImageLoader

from pdf_appRecognizer.classes.converter import Converter
from pdf_appRecognizer.classes.pdf import PdfCamelot, PdfTextReader
from pdf_appRecognizer.classes.data_json import DataCleaning, DataCollection, DictToJson
from pdf_appRecognizer.classes.text_extracter import DataExtractionImage, DataExtractionPDF
from pdf_appRecognizer.classes.img import ImageDataExtracter, tesseract_languages


load_dotenv()
TESSERACT_OCR: str = os.getenv('TESSERACT')


def improve_img_quality(img_path: str, output_path: str, sharpness: int = 1, contrast: float = 1.3,
                        blur: int = 1) -> None:
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
    pil_img = Img.fromarray(img)

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

    # Конвертируем в PIL Image (Im) и сохраняем
    img_enhanced = Img.fromarray(img_enhanced)
    img_enhanced.save(output_path)


def upscale_image(path_to_based_img: str, path_to_upscaled_img: str, model: torch.nn.Module) -> None:
    """
    Функция апскейла (улучшение качества изображения/увеличение разрашения изображения)
    :param path_to_based_img: Путь до изображения, которое будем улучшать
    :param path_to_upscaled_img: Сохранение улучшенной картинки (формат: название_папки/название_картинки)
    :param model: Выбор нейронной модели, для прогона изображения
    :return: None
    """

    cur_device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = model.to(cur_device)

    img = np.array(Img.open(path_to_based_img), dtype=np.float32) / 255.0
    img = img[:, :, 0:3]

    tile_count_x: int = 16
    tile_count_y: int = 16

    m = img.shape[0] // tile_count_x
    n = img.shape[1] // tile_count_y

    tiles = [[img[x:x + m, y:y + n] for x in range(0, img.shape[0], m)] for y in range(0, img.shape[1], n)]
    inputs = [[torch.from_numpy(tile).permute(2, 0, 1).unsqueeze(0).to(cur_device) for tile in part] for part in tiles]

    upscaled = None
    count: int = 0

    for i in range(tile_count_y + 1):
        col = None
        for j in range(tile_count_x + 1):
            pred = model(inputs[i][j])
            res = pred.detach().to('cpu').squeeze(0).permute(1, 2, 0)
            # print(f"Image tile #{count}. Upscaled shape: {res.shape}")
            count += 1
            col = res if col is None else torch.cat([col, res], dim=0)
            del pred
        upscaled = col if upscaled is None else torch.cat([upscaled, col], dim=1)

    # Сохраняем итоговое изображение
    cv2.imwrite(fr'pdf_appRecognizer/extract_assets/image_files/{path_to_upscaled_img}',
                upscaled.numpy() * 255.0)

    torch.cuda.empty_cache()


def dt_img2excel(img_path: str, xlsx_path: str, ocr_language: str = 'rus') -> None:
    ocr = TesseractOCR(lang=ocr_language)
    # Создаем экзмепляр класса документ
    doc_dt = Image(src=img_path)
    extracted_tables = doc_dt.extract_tables(ocr=ocr)

    table_img = cv2.imread(filename=img_path)

    for table in extracted_tables:
        for row in table.content.values():
            for cell in row:
                cv2.rectangle(table_img, (cell.bbox.x1, cell.bbox.y1), (cell.bbox.x2, cell.bbox.y2), (255, 0, 0), 2)

    doc_dt.to_xlsx(dest=xlsx_path, ocr=ocr)


def prep_image_var(input_img: str, output_img: str) -> None:
    """
    Эта функция выделяет именно табличную часть (тестировал только для одностраничных УПД)
    :return: None
    """
    img = cv2.imread(input_img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    # Выделяем контуры для табличной части
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Перебор найденных контуров
    for cnt in contours:
        area = cv2.contourArea(cnt)

        # Фильтрация по площади контура
        if area > 1000:  # Пороговое значение для фильтрации
            # Создание маски для выделения контура
            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.drawContours(mask, [cnt], -1, (255), thickness=cv2.FILLED)

            # Записываем новое изображение, где исключительно табличная часть
            table_image = cv2.bitwise_and(img, img, mask=mask)
            cv2.imwrite(output_img, img=table_image)
            return


            # Извлечение области таблицы
            table_image_2 = cv2.imread('pdf_appRecognizer/extract_assets/image_files/Test_Table_YPD_2.png')
            gray_img = cv2.cvtColor(table_image_2, cv2.COLOR_BGR2GRAY)
            cv2.imwrite('pdf_appRecognizer/extract_assets/image_files/Test_Table_YPD_2_2.png', img=gray_img)


def extract_dt_img(img_file: str):
    image = cv2.imread(filename=img_file)

    config_options = r'--psm 6'
    data = pytesseract.image_to_string(image=image, config=config_options, lang='rus')
    return data


def data_write_collection(collection: dict, filename: str,
                          path: str = 'pdf_appRecognizer/extract_assets/json_files') -> None:
    # функция для записи извлеченных значений из изображения в формате .json (НЕ табличная часть)
    """

    :param filename: наименование файла в формате .json
    :param collection: Хэш-таблица/словарь
    :param path: Путь сохранения json файла
    :return:
    """
    try:
        with open(file=f'{path}/{filename}.json', mode='w', encoding='utf-8') as json_file:
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

    Converter.from_png_to_pdf(png_path='pdf_appRecognizer/extract_assets/image_files/UPD_1.png',
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


def image_extracting(image_file_and_folder: str, image_lang: str):
    """
    Эта функция через регулярки извлекает данные ВНЕ таблицы. Работает по НЕСТРУКТУРИРОВАННОМУ тексту
    :param image_lang: Язык, который нужно извлечь из картинки
    :param image_file_and_folder: Путь до файла (Формат: название_папки/название_файла)
    :return: None
    """

    # Создаем экземпляр класса curr_image для работы с ним. Структура таблицы в таком варианте теряется.
    curr_image = ImageDataExtracter(path_dir='pdf_appRecognizer/extract_assets/image_files',
                                    image_file=image_file_and_folder,
                                    path_to_tesseract=TESSERACT_OCR, language=image_lang)

    extracted_text = curr_image.tesseract_extraction()
    # Вывод извлеченного текста (без структуры)
    print(extracted_text, '\n\n')
    return

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


def test():
    image_extracting(image_file_and_folder='YPD_1/UPD_1.png',
                     image_lang='rus')    # Извлекаем из базовой картинки
    #
    improve_img_quality(img_path='pdf_appRecognizer/extract_assets/image_files/YPD_1/enhanced_UPD_1_scale_2.png',
                        output_path='pdf_appRecognizer/extract_assets/image_files/YPD_1/enhanced_2x_UPD_1_scale_2.png')

    print('---------------------------------------------------------------------\n\n')

    # img_path = r'pdf_appRecognizer/extract_assets/image_files/YPD_1/enhanced_UPD_1.png'
    #
    # cur_model = EdsrModel.from_pretrained('eugenesiow/edsr-base', scale=2)

    # upscale_image(path_to_based_img=img_path, path_to_upscaled_img='YPD_1/enhanced_UPD_1_scale_2.png', model=cur_model)
    #

    image_extracting(image_file_and_folder='YPD_1/enhanced_2x_UPD_1_scale_2.png',
                     image_lang='rus')    # Извлекаем из базовой картинки

    # dt_img2excel(img_path='pdf_appRecognizer/extract_assets/image_files/YPD_1/enhanced_UPD_1.png',
    #              xlsx_path='pdf_appRecognizer/extract_assets/xlsx_files/test_3.xlsx')

    dt_img2excel(img_path='pdf_appRecognizer/extract_assets/image_files/YPD_1/enhanced_2x_UPD_1_scale_2.png',
                 xlsx_path='pdf_appRecognizer/extract_assets/xlsx_files/test_4.xlsx')


test()
