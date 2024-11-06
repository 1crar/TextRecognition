import PIL.ImageFilter
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
from pytesseract import Output
from PIL import Image, ImageEnhance
from super_image import EdsrModel, ImageLoader

from pdf_appRecognizer.classes.converter import Converter
from pdf_appRecognizer.classes.pdf import PdfCamelot, PdfTextReader
from pdf_appRecognizer.classes.data_json import DataCleaning, DataCollection, DictToJson
from pdf_appRecognizer.classes.text_extracter import DataExtractionImage, DataExtractionPDF
from pdf_appRecognizer.classes.img import ImageDataExtracter, tesseract_languages


load_dotenv()
TESSERACT_OCR: str = os.getenv('TESSERACT')


def upscale(img_path: str, model: torch.nn.Module):
    curr_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.cuda.empty_cache()

    model = model.to(curr_device)

    img = np.array(Image.open(img_path), dtype=np.float32) / 255.0
    img = img[:,:,0:3]
    tileCountX = 16
    tileCountY = 16
    M = img.shape[0] // tileCountX
    N = img.shape[1] // tileCountY
    tiles = [[img[x:x+M, y:y+N] for x in range(0, img.shape[0], M)] for y in range(0, img.shape[1], N)]
    inputs = [[torch.from_numpy(tile).permute(2, 0, 1).unsqueeze(0).to(curr_device) for tile in part] for part in tiles]

    upscaled_img = None
    count = 0

    for i in range(tileCountY + 1):
        col = None
        for j in range(tileCountX + 1):
            pred = model(inputs[i][j])
            res = pred.detach().to('cpu').squeeze(0).permute(1, 2, 0)
            print(f"Image tile #{count}. Upscaled shape: {res.shape}")
            count += 1
            col = res if col is None else torch.cat([col, res], dim=0)
            del pred
        upscaled_img = col if upscaled_img is None else torch.cat([upscaled_img, col], dim=1)

    f, axarr = plt.subplots(1,2)
    axarr[0].imshow(upscaled_img)
    axarr[1].imshow(img)

    plt.show()

    cv2.imwrite(img_path.split('\\')[-1].split('.')[0] + "_4x.png", upscaled_img.numpy() * 255.0)


def improve_img_quality(img_path: str, output_path: str, sharpness: int = 1, contrast: float = 1.3, blur: int = 1):
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

    # # Переде увеличением резкости, увеличим резекость методом SHARPEN класса ImageFilter
    # sharp_image = pil_img.filter(PIL.ImageFilter.SHARPEN)

    # # Улучшаем качество границ символов на текущем изображении
    # edge_img = sharp_image.filter(PIL.ImageFilter.EDGE_ENHANCE)
    #
    # # Используем метод EDGE_ENHANCE_MORE для еще более существенного улучшения границ символов
    # edge_img_enhanced = edge_img.filter(PIL.ImageFilter.EDGE_ENHANCE_MORE)

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


def prep_image_var(input_img: str, output_img: str):
    """
    Эта функция выделяет именно табличную часть. Но тестировал только для одностраничных УПД
    :return: None
    """
    img = cv2.imread(input_img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, binary = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY_INV)
    # Выделяем контуры для табличной части
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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


def image_extracting(image_file: str, image_lang: str):
    """
    Эта функция через регулярки извлекает данные ВНЕ таблицы. Работает по НЕСТРУКТУРИРОВАННОМУ тексту
    :param image_lang: Язык, который нужно извлечь из картинки
    :param image_file: Путь до файла
    :return: None
    """

    # Создаем экземпляр класса curr_image для работы с ним. Структура таблицы в таком варианте теряется.
    curr_image = ImageDataExtracter(path_dir='pdf_appRecognizer/extract_assets/image_files',
                                    image_file=image_file,
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


"""
А далее уже я тестировал функции на качество извлечения и улучшения изображений.
"""


# prep_image_var(input_img='pdf_appRecognizer/extract_assets/image_files/Nex_enhanced_YPD_x4.jpg',
#                output_img='pdf_appRecognizer/extract_assets/image_files/New_enhanced_YPD_x4_table.jpg')
#
# dt = extract_dt_img(img_file='pdf_appRecognizer/extract_assets/image_files/New_enhanced_YPD_x4_table.jpg')
# print(dt)


# improve_img_quality(img_path='pdf_appRecognizer/extract_assets/image_files/Test_Table_YPD_2.png',
#                     output_path='pdf_appRecognizer/extract_assets/image_files/enhanced_Test_Table_YPD_2.png')

# table2img(path_to_img='pdf_appRecognizer/extract_assets/image_files/enhanced_Test_Table_YPD_2.png')


def test():
    # img_path: str = 'pdf_appRecognizer/extract_assets/image_files/Test_Table_YPD_2.png'

    image_extracting(image_file='Test_Table_YPD_2.png', image_lang='rus')

    # improve_img_quality(img_path='pdf_appRecognizer/extract_assets/image_files/Test_Table_YPD_2.png',
    #                     output_path='pdf_appRecognizer/extract_assets/image_files/enhanced_Test_Table_YPD_2.png')

    print('---------------------------------------------------------------------\n\n')

    image_extracting(image_file='Test_Table_YPD_2_scale_x4.png', image_lang='rus')

    # image_file: str = 'pdf_appRecognizer/extract_assets/image_files/enhanced_english_x2.png'
    #
    # with Image.open(fp=image_file) as curr_img:
    #     sharp_img = curr_img.filter(PIL.ImageFilter.SHARPEN)
    #
    #     for _ in range(10):
    #         sharp_img = sharp_img.filter(PIL.ImageFilter.SHARPEN)
    #
    #     for _ in range(1):
    #         sharp_img = curr_img.filter(PIL.ImageFilter.EDGE_ENHANCE)
    #
    #     for _ in range(1):
    #         sharp_img = sharp_img.filter(PIL.ImageFilter.EDGE_ENHANCE_MORE)
    #
    #     sharp_img.show()


        # for i in range(15):
        #     edges = sharp_img.filter(PIL.ImageFilter.EDGE_ENHANCE)      # EDGE_ENHANCE
        # edges.show()
        # edges.save(fp='pdf_appRecognizer/extract_assets/image_files/edgy_enhanced_english_x3.png')



    # tess_lang = tesseract_languages(path_to_tesseract=TESSERACT_OCR)
    #
    # tess_dict: dict = {
    #     "lang_list": tess_lang
    # }
    # data_write_collection(collection=tess_dict, filename='tesseract_languages')
    #
    # print(tess_lang)


test()
