import cv2
import os

from dotenv import load_dotenv
from img2table.document import Image
from PIL import Image, ImageDraw, ImageFont

import torch
import easyocr
import numpy as np

from img2table.ocr import EasyOCR
from PIL import Image as Img

load_dotenv()


image_path: str = f'temp/test_4_final_result_x2.png'
#
image_cv2 = cv2.imread(filename=image_path)
languages: list = ['ru']
#
# dt_img2excel(img_path=image_path)
#
#
print("[INFO] OCR'ing input image...")
print(f"[INFO] OCR'ing with the following languages: {languages}")
#
test_case: int = 2
results: list = []
#
if test_case == 1:
    # reader = easyocr.Reader(lang_list=languages, gpu=True,
    #                         model_storage_directory='fine_tune', user_network_directory='fine_tune',
    #                         recog_network='ru_finetune')
    reader = easyocr.Reader(lang_list=languages, gpu=True)
    # results = reader.readtext(image_cv2, text_threshold=0.75, contrast_ths=0.7, width_ths=1.25, height_ths=0.75,
    #                           ycenter_ths=0.5, slope_ths=1, add_margin=0.175, decoder='wordbeamsearch', beamWidth=20,
    #                           canvas_size=3500)
    # Not bad
    results = reader.readtext(image_cv2, text_threshold=0.7, contrast_ths=0.8, width_ths=1.2, height_ths=0.8,
                              ycenter_ths=0.5, slope_ths=1, add_margin=0.200, decoder='wordbeamsearch', beamWidth=20,
                              canvas_size=3500)
if test_case == 2:
    reader = easyocr.Reader(lang_list=languages, gpu=True)
    # results = reader.readtext(image_cv2)
    results = reader.readtext(image_cv2)

# print(results)
#
# # Преобразуем изображение OpenCV в формат PIL
# image_pil = Image.fromarray(cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB))
# draw = ImageDraw.Draw(image_pil)
#
# # Выберите шрифт (если нужный шрифт не установлен, вы можете указать путь к .ttf файлу)
# font = ImageFont.truetype(font="arial.ttf", size=20)  # Убедитесь, что путь к шрифту корректный
#
for (bbox, text, prob) in results:
    print("[INFO] {:.4f}: {}".format(prob, text))
    # Распаковка координат
    (tl, tr, br, bl) = bbox
    tl = (int(tl[0]), int(tl[1]))
    br = (int(br[0]), int(br[1]))
#
#     # Рисуем прямоугольник
#     draw.rectangle([tl, br], outline=(0, 255, 0), width=2)
#
#     # Рисуем текст
#     draw.text((tl[0], tl[1] - 10), text, fill=(0, 255, 0), font=font)
#
# # Отображаем результат
# image_pil.show()


class TableExtracter:
    def __init__(self, image_path: str):
        self.image_path = image_path

        self.gray_img = None
        self.denoised_image = None
        self.contours = None
        # self.dt_image = None
        self.rectangular_contours = []

    def image_processing(self, is_debugging: bool = True):
        """Обрабатывает изображение: преобразует в черно-белое, инвертирует и проводит морфологические операции."""
        # Считываем изображение и преобразуем в np.array
        img = cv2.imread(filename=self.image_path)

        if img is None:
            raise ValueError(f"Не удалось загрузить изображение из {self.image_path}")

        # Делаем серым
        self.gray_img = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)
        # Уменьшаем изображение до черных и белых пикселей (порог)
        thresh_hold_img = cv2.threshold(self.gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        # Инвертируем изображения для последующих операций
        inverted_image = cv2.bitwise_not(thresh_hold_img)

        # Удаление шумов и морфологические операции
        blurred_image = cv2.GaussianBlur(inverted_image, (1, 1), 0)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))

        denoised_image = cv2.morphologyEx(blurred_image, cv2.MORPH_CLOSE, kernel)
        self.denoised_image = cv2.morphologyEx(denoised_image, cv2.MORPH_OPEN, kernel)

        # Уплотняем контуры

        if is_debugging:
            cv2.imwrite(filename='temp/test.png', img=self.denoised_image)

        # return self.denoised_image

    def find_contours(self, is_debugging: bool = True):
        contours, _ = cv2.findContours(self.denoised_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if is_debugging:
            image_with_all_contours = self.gray_img.copy()
            cv2.drawContours(image_with_all_contours, contours, -1, (0, 255, 0), 2)
            cv2.imwrite(filename='temp/test_2.png', img=image_with_all_contours)

        self.contours = contours
        # return self.contours

    def filter_contours_and_leave_only_rectangles(self, index: float, is_debugging: bool = True):
        for contour in self.contours:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, index * peri, True)

            if len(approx) == 4:
                self.rectangular_contours.append(approx)

        if is_debugging:
            image_with_only_rectangular_contours = self.gray_img.copy()
            cv2.drawContours(image_with_only_rectangular_contours, self.rectangular_contours, -1, (0, 255, 0), 3)
            cv2.imwrite(filename='temp/test_3.png', img=image_with_only_rectangular_contours)

        return self.rectangular_contours

    def crop_rectangles_to_single_image(self, dt_coloured_img: str, min_area=4000, is_debugging: bool = True):
        img = cv2.imread(filename=dt_coloured_img)

        debug_image = None
        if is_debugging:
            debug_image = img.copy()

        # Инициализация крайних координат
        min_x, min_y = float('inf'), float('inf')
        max_x, max_y = float('-inf'), float('-inf')

        # Обработка прямоугольников для нахождения крайних точек
        for rect in self.rectangular_contours:
            x, y, w, h = cv2.boundingRect(rect)
            area = w * h
            if area >= min_area:
                # Обновляем крайние координаты
                min_x = min(min_x, x)
                min_y = min(min_y, y)
                max_x = max(max_x, x + w)
                max_y = max(max_y, y + h)

                if is_debugging:
                    # Рисуем прямоугольник на отладочном изображении
                    cv2.rectangle(debug_image, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Проверка, были ли найдены подходящие прямоугольники
        if min_x == float('inf') or min_y == float('inf') or max_x == float('-inf') or max_y == float('-inf'):
            raise ValueError("Не найдено ни одного подходящего прямоугольника с достаточной площадью.")

        # Обрезаем изображение по найденным крайним координатам
        cropped_image = img[min_y:max_y, min_x:max_x]

        if is_debugging:
            cv2.rectangle(debug_image, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)  # Рисуем общий прямоугольник
            cv2.imwrite(filename='temp/test_combined.png', img=debug_image)

        cv2.imwrite(filename='temp/test_4.png', img=cropped_image)
        return cropped_image


def cropped_based_img(input_file: str, output_file: str):
    img = cv2.imread(filename=input_file)
    height_img = img.shape[0]

    # Определение высоты для обрезки (верхняя половина)
    y1 = round(height_img - height_img * 0.75)
    # Определение высоты для обрезки (нижняя половина)
    y2 = round(height_img * 0.75)

    # print(img.shape, height_img, sep='\n')

    cropped_img = img[y1:y2, :]
    cv2.imwrite(filename=output_file, img=cropped_img)


def cropped_img_files(folder_path: str):
    for img_file in os.listdir(path=folder_path):
        output_file_name: str = ''

        if img_file.endswith('.png'):
            output_file_name = f'{img_file.replace('.png', '')}_cropped.png'
        if img_file.endswith('.jpg'):
            output_file_name = f'{img_file.replace('.jpg', '')}_cropped.jpg'

        cropped_based_img(input_file=f'{folder_path}{img_file}', output_file=f'{folder_path}{output_file_name}')


# curr_folder_path: str = 'pdf_appRecognizer/extract_assets/image_files/YPDs/trash/'
# cropped_img_files(folder_path=curr_folder_path)


def find_contours(dilated_img, is_debugging: bool = True):
    local_dilated_image = dilated_img
    cur_contours, cur_hierarchy = cv2.findContours(local_dilated_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if is_debugging:
        image_with_all_contours = gray_img.copy()
        cv2.drawContours(image_with_all_contours, cur_contours, -1, (0, 255, 0), 2)
        cv2.imwrite(filename='temp/test_2.png', img=image_with_all_contours)

    return cur_contours, cur_hierarchy


def filter_contours_and_leave_only_rectangles(contours, is_debugging: bool = True):
    rectangular_contours = []

    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.01 * peri, True)

        if len(approx) == 4:
            rectangular_contours.append(approx)

    if is_debugging:
        image_with_only_rectangular_contours = gray_img.copy()
        cv2.drawContours(image_with_only_rectangular_contours, rectangular_contours, -1, (0, 255, 0), 3)
        cv2.imwrite(filename='temp/test_3.png', img=image_with_only_rectangular_contours)
    return rectangular_contours


def crop_rectangles_to_single_image(image_path, rectangles, min_area=3000, is_debugging: bool = True):
    # Читаем изображение из файла
    img = cv2.imread(filename=image_path)

    # Проверка на успешное чтение изображения
    if img is None:
        raise ValueError(f"Не удалось загрузить изображение из {image_path}")

    # Определяем цвета для отладки
    debug_image = None
    if is_debugging:
        debug_image = img.copy()

    # Инициализация крайних координат
    min_x, min_y = float('inf'), float('inf')
    max_x, max_y = float('-inf'), float('-inf')

    # Обработка прямоугольников для нахождения крайних точек
    for rect in rectangles:
        x, y, w, h = cv2.boundingRect(rect)
        area = w * h
        if area >= min_area:
            # Обновляем крайние координаты
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x + w)
            max_y = max(max_y, y + h)

            if is_debugging:
                cv2.rectangle(debug_image, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Рисуем прямоугольник на отладочном изображении

    # Проверка, были ли найдены подходящие прямоугольники
    if min_x == float('inf') or min_y == float('inf') or max_x == float('-inf') or max_y == float('-inf'):
        raise ValueError("Не найдено ни одного подходящего прямоугольника с достаточной площадью.")

    # Обрезаем изображение по найденным крайним координатам
    cropped_image = img[min_y:max_y, min_x:max_x]

    if is_debugging:
        cv2.rectangle(debug_image, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)  # Рисуем общий прямоугольник
        cv2.imwrite(filename='temp/test_combined.png', img=debug_image)

    return cropped_image


# test_img_path: str = 'temp/test_11_denoised_dt.png'
# test_img = cv2.imread(filename=test_img_path)
# # Шаг 1 - делаем серым
# gray_img = cv2.cvtColor(src=test_img, code=cv2.COLOR_BGR2GRAY)
# # Шаг 2 - Уменьшаем изображение доа черных и белых пикселей (порог от белого до черного пикселя)
# thresh_hold_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
# # Шаг 3 - Инвертируем изображения для последующих операций
# inverted_image = cv2.bitwise_not(thresh_hold_img)
# # Шаг 4 - С помощью методов dilate и erode происходит утолщение всех линий. Поможет определить контуры далее.
# # erode_image = cv2.erode(inverted_image, None, iterations=1)
#
# blurred_image = cv2.GaussianBlur(inverted_image, (1, 1), 0)
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
#
# denoised_image = cv2.morphologyEx(blurred_image, cv2.MORPH_CLOSE, kernel)
# denoised_image = cv2.morphologyEx(denoised_image, cv2.MORPH_OPEN, kernel)
#
# # dilated_image = cv2.dilate(denoised_image, None, iterations=1)
#
# cv2.imwrite(filename='temp/test.png', img=denoised_image)
#
# # Находим контуры
# img_contours, _ = find_contours(dilated_img=denoised_image)
# # Из найденных контуров извлекаем только прямоугольники
# cur_rectangular_contours = filter_contours_and_leave_only_rectangles(contours=img_contours)
# # Найденные прямоугольники накладываем на базовое изображение
# dt_image = crop_rectangles_to_single_image(
#         image_path='pdf_appRecognizer/extract_assets/image_files/YPDs/trash/24_cropped.png',
#         rectangles=cur_rectangular_contours,
#         min_area=4000)
# cv2.imwrite(filename='temp/test_4.png', img=dt_image)


def test():
    img_path: str = 'pdf_appRecognizer/extract_assets/image_files/YPDs/trash/'
    coloured_img_file: str = '24_cropped.png'
    uncoloured_img_file: str = '24_uncoloured_cropped.png'

    img_instance = TableExtracter(image_path=f'{img_path}{uncoloured_img_file}')

    img_instance.image_processing()
    img_instance.find_contours()

    img_instance.filter_contours_and_leave_only_rectangles(index=0.01)
    # возвращаем не путь до обрезанного изображения, а np.array
    dt_image = img_instance.crop_rectangles_to_single_image(dt_coloured_img=f'{img_path}{coloured_img_file}')
    print(dt_image, type(dt_image), sep='\n')


# test()
