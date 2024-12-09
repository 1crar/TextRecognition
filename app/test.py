import copy

import cv2
import easyocr
import json
import os
import torch
import pytesseract

import pandas as pd
import numpy as np

from dotenv import load_dotenv
from img2table.document import Image
from img2table.ocr import TesseractOCR, EasyOCR
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


def improve_img_quality(img_path: str, output_path: str, sharpness: int = 1, contrast: float = 0.7,
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
    # Уплотняем шрифт
    img = cv2.erode(src=img, kernel=np.ones((2, 2)), iterations=1)

    # Придаем серый оттеннок для лучшего распознавания TesseractOCR
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # конвертируем image в PIL Image
    pil_img = Img.fromarray(gray_img)

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
    img_enhanced.show()
    img_enhanced.save(output_path)


def upscale_image(path_to_based_img: str, path_to_upscaled_img: str, model: torch.nn.Module) -> str:
    """
    Функция апскейла (улучшение качества изображения/увеличение разрашения изображения)
    :param path_to_based_img: Путь до изображения, которое будем улучшать
    :param path_to_upscaled_img: Сохранение улучшенной картинки (формат: название_папки/название_картинки)
    :param model: Выбор нейронной модели, для прогона изображения
    :return: None
    """

    cur_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(cur_device)

    # Конвертация обратно в "RGB" для создания 3-измерения (модель прогоняет только их)
    img = np.array(Img.open(path_to_based_img).convert("RGB"), dtype=np.float32) / 255.0

    tile_count_x: int = 16
    tile_count_y: int = 16

    upscaled = None
    count: int = 0

    # Проверка 3-мерный массив
    if len(img.shape) == 3:
        img = img[:, :, 0:3]

        m = img.shape[0] // tile_count_x
        n = img.shape[1] // tile_count_y

        tiles = [[img[x:x + m, y:y + n] for x in range(0, img.shape[0], m)] for y in range(0, img.shape[1], n)]
        inputs = [[torch.from_numpy(tile).permute(2, 0, 1).unsqueeze(0).to(cur_device) for tile in part] for part in
                  tiles]
        for i in range(tile_count_y):
            col = None
            for j in range(tile_count_x):
                pred = model(inputs[i][j])
                res = pred.detach().to('cpu').squeeze(0).permute(1, 2, 0)
                # print(f"Image tile #{count}. Upscaled shape: {res.shape}")
                count += 1
                col = res if col is None else torch.cat([col, res], dim=0)
                del pred
            upscaled = col if upscaled is None else torch.cat([upscaled, col], dim=1)
    # Сохраняем итоговое изображение
    cv2.imwrite(path_to_upscaled_img, upscaled.numpy() * 255.0)
    torch.cuda.empty_cache()

    return path_to_upscaled_img


def upscale_img_ver2(path_to_based_img: str, path_to_upscaled_img: str, model: torch.nn.Module) -> str:
    cur_device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = model.to(cur_device)

    # Освободим кэш памяти CUDA
    torch.cuda.empty_cache()

    # Предобработка изображения
    img = np.array(Img.open(path_to_based_img))

    # Уплотняем шрифт
    img = cv2.erode(src=img, kernel=np.ones((2, 2)), iterations=1)

    # Изменение размера для уменьшения использования памяти
    # img = cv2.resize(img, (img.shape[1] // 4, img.shape[0] // 4))

    # Обратная трансформация изображения из np.array
    img = Img.fromarray(img)
    img.show()
    img.save(fp='temp/enhanced_test_2.png')

    inputs = ImageLoader.load_image(image=img)
    inputs = inputs.to(cur_device)

    preds = model(inputs)

    ImageLoader.save_image(pred=preds, output_file=path_to_upscaled_img)


def detect_datatable_part(ypd_img_path: str, output_filename: str, temp_filename: str, offset: int = 0,
                          is_erode: bool = True) -> str:
    ocr_detector = EasyOCR(lang=['ru'])
    # Создаем экземпляр класса документ
    doc_dt = Image(src=ypd_img_path)
    extracted_tables = doc_dt.extract_tables(ocr=ocr_detector)

    table_img = cv2.imread(filename=ypd_img_path)
    output_path: str = output_filename

    for idx, table in enumerate(extracted_tables):
        # Инициализируем минимальные и максимальные границы
        min_x1 = min(cell.bbox.x1 for row in table.content.values() for cell in row)
        min_y1 = min(cell.bbox.y1 for row in table.content.values() for cell in row)
        max_x2 = max(cell.bbox.x2 for row in table.content.values() for cell in row)
        max_y2 = max(cell.bbox.y2 for row in table.content.values() for cell in row)

        # Вычисляем высоту таблицы и соотношение для нижнего отступа (так как иногда не захватывается ласт строка)
        table_height = max_y2 - min_y1
        # Динамически будет создаваться отступ в пропорции 5% от всей высоты таблицы
        dynamic_lower_offset = int(table_height * 0.05)

        # Создаем общий отступ
        total_y_min = max(0, min_y1 - offset)
        total_y_max = max_y2 + offset + dynamic_lower_offset

        # Обрезаем всю табличную часть с учетом границ и отступа
        crop_dt_img = table_img[
                      total_y_min:total_y_max,
                      max(0, min_x1 - offset):max_x2 + offset
                      ]
        # Следующие строки нужны для отладки процесса распознавания табличной части
        detected_lines = copy.deepcopy(table_img)
        # Прорисовка границ табличной части
        for row in table.content.values():
            for cell in row:
                x1, y1, x2, y2 = cell.bbox.x1, cell.bbox.y1, cell.bbox.x2, cell.bbox.y2
                cv2.rectangle(detected_lines, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.imwrite(filename=temp_filename, img=detected_lines)

        # Сохраняем изображение табличной части
        try:
            if is_erode:
                crop_dt_img = cv2.erode(crop_dt_img, kernel=np.ones((2, 2)), iterations=1)
            cv2.imwrite(output_path, crop_dt_img)
        except Exception as e:
            # print(f"ERROR: {e}")
            raise Exception(f"ERROR: ошибка записи файла - {e}")

    return output_path


def dt_img_extracter(img_path: str, offset: int = 3, is_cell_remove: bool = True) -> list:
    """
    :param is_cell_remove:
    :param is_erode:
    :param img_path:
    :param offset:
    :return:
    """
    ocr = EasyOCR(lang=['ru'])
    print(ocr.reader.readtext(image=img_path))

    # Создаем экземпляр класса документ
    doc_dt = Image(src=img_path)
    extracted_tables = doc_dt.extract_tables(ocr=ocr)

    table_img = cv2.imread(filename=img_path)
    extracted_data: list = []

    # Обрезаем по каждой ячейке
    for idx, table in enumerate(extracted_tables):
        for row in table.content.values():
            for cell in row:
                # Прорисовка табличной части
                cv2.rectangle(table_img, (cell.bbox.x1, cell.bbox.y1), (cell.bbox.x2, cell.bbox.y2), (255, 0, 1), 2)
                # Обрезаем область ячейки + пиксельный отступ
                crop_img = table_img[
                           max(0, cell.bbox.y1 + offset):cell.bbox.y2 - offset,
                           max(0, cell.bbox.x1 + offset):cell.bbox.x2 - offset
                           ]
                # Уплотнение шрифта
                # if is_erode:
                #     crop_img = cv2.erode(crop_img, kernel=np.ones((2, 2)), iterations=1)
                try:
                    temp_cell_path_img: str = (f'pdf_appRecognizer/extract_assets/image_files/dt_cells/'
                                               f'table_cell_{idx}_{cell.bbox.x1}_{cell.bbox.y1}.png')
                    cv2.imwrite(temp_cell_path_img, crop_img)

                    # Инициализируем объект reader (EasyOCR)
                    reader = easyocr.Reader(lang_list=['ru'], gpu=True)
                    results = reader.readtext(temp_cell_path_img, text_threshold=0.7, contrast_ths=0.8, width_ths=1.2,
                                              height_ths=0.8, ycenter_ths=0.5, slope_ths=1, add_margin=0.200,
                                              decoder='wordbeamsearch', beamWidth=20, canvas_size=3500)
                    # Проходимся по циклу и добавляем извлеченное EasyOCR значение
                    for el_tuple in results:
                        print(f'INFO: OCR\'s extracted datatable value: {el_tuple[1]}')
                        extracted_data.append(el_tuple[1])
                    # Если True - удаляем вырезанную ячейку
                    if is_cell_remove:
                        os.remove(path=temp_cell_path_img)

                except Exception as e:
                    print(f"ERROR: {e}")

    Img.fromarray(table_img).save("temp/test_scaled_4.png")
    # doc_dt.to_xlsx(dest=xlsx_path, ocr=ocr)
    return extracted_data


def detect_dt_part(input_img: str, output_img: str) -> None:
    """
    Эта функция выделяет именно табличную часть (тестировал только для одностраничных УПД)
    :return: None
    """
    img = cv2.imread(input_img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # (Изображение уже серое)

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


def image_extracting(image_file_and_folder: str, image_lang: str, is_tesseract: bool = False,
                     is_paragraph: bool = False):
    """
    Эта функция через регулярки извлекает данные ВНЕ таблицы. Работает по НЕСТРУКТУРИРОВАННОМУ тексту
    :param is_paragraph:
    :param is_tesseract: параметр, который обозначает использование TesseractOCR
    :param image_lang:
    :param image_file_and_folder: Путь до файла (Формат: название_папки/название_файла)
    :return: None
    """

    if is_tesseract:
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
    else:
        reader = easyocr.Reader([image_lang[:len(image_lang)-1]])
        img = Img.open(fp=f'pdf_appRecognizer/extract_assets/image_files/{image_file_and_folder}')

        img = img.convert('L')     # Преобразуем в серое изображение
        # Эта строчка нужна для преобразования в numpy array
        gray_img = cv2.imread(filename=f'pdf_appRecognizer/extract_assets/image_files/{image_file_and_folder}')

        allow_list: str = 'АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ-'

        result = reader.readtext(gray_img, detail=0, paragraph=is_paragraph,
                                 allowlist=f'{allow_list}/{allow_list[:len(allow_list)-1].lower()}')
        print(result)


def test(is_img_processing: bool = False):
    # image_extracting(image_file_and_folder='YPD_3/YPD_3.png',
    #                  image_lang='rus', is_paragraph=False)    # Извлекаем из базовой картинки

    print('---------------------------------------------------------------------\n\n')

    if is_img_processing:
        # img_path = r'pdf_appRecognizer/extract_assets/image_files/YPD_4/YPD_4.png'
        #
        # cur_model = EdsrModel.from_pretrained('eugenesiow/edsr-base', scale=2)
        #
        # upscale_image(path_to_based_img=img_path, path_to_upscaled_img='YPD_4/YPD_4_scales_2.png',
        #               model=cur_model)

        improve_img_quality(img_path='pdf_appRecognizer/extract_assets/image_files/YPD_1/UPD_1.png',
                            output_path='pdf_appRecognizer/extract_assets/image_files/YPD_1/enhanced_UPD_1.png')

    # image_extracting(image_file_and_folder='YPD_3/enhanced_YPD_3_scales_2.png',
    #                  image_lang='rus', is_paragraph=False)    # Извлекаем из базовой картинки

    # is_paragraph=True нужен только для работы с таблицами данных.Данный параметр лучше сегментирует данные.
    #
    # dt_img2excel(img_path='pdf_appRecognizer/extract_assets/image_files/YPD_3/YPD_3_scales_2.png',
    #              xlsx_path='pdf_appRecognizer/extract_assets/xlsx_files/test_7.xlsx', is_tesseract=False)


# test(is_img_processing=True)


# based_img_path: str = 'pdf_appRecognizer/extract_assets/image_files/YPDs/trash/76.jpg'
#
# dt_img: str = detect_datatable_part(ypd_img_path=based_img_path,
#                                     output_filename=based_img_path.replace('34.jpg',
#                                                                            '34_dt_part.jpg'),
#                                     temp_filename='temp/test_34_dt_part.jpg',
#                                     offset=0,
#                                     is_erode=False)

# cur_model = DrlnModel.from_pretrained('eugenesiow/drln', scale=4)
# # upscale_image() возвращает путь до upscaled_x4 табличного изображения
# dt_upscaled_img = upscale_image(path_to_based_img=dt_img,
#                                 path_to_upscaled_img=dt_img.replace('26_dt_part.jpg',
#                                                                     'upscaled_4_26_dt_part.jpg'),
#                                 model=cur_model)
#
# data = dt_img_extracter(img_path='temp/26_dt_part.jpg')

# print(data)
# dt_collection: dict = {
#     "datatable": data
# }
# # Запись извлеченных табличных данных в json.
# DictToJson.write_to_json(collection=dt_collection, path_to_save='temp/temp_dt_jsons/upscaled_4_26_dt_part.json')


# detect_dt_part(input_img=temp, output_img=temp.replace('test_3.png', 'test_5.png'))

img_path = 'pdf_appRecognizer/extract_assets/image_files/YPDs/trash/24_cropped.png'
improve_img_quality(img_path=img_path, output_path=img_path, sharpness=14, contrast=3, blur=1)
