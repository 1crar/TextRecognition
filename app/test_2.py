import cv2

from dotenv import load_dotenv
from img2table.document import Image
from PIL import Image, ImageDraw, ImageFont

import torch
import easyocr
import numpy as np

from img2table.ocr import EasyOCR
from PIL import Image as Img

load_dotenv()


# # enhanced_file: str = 'temp/scaled4_loveimg_test_2.png'
# image_path: str = f'pdf_appRecognizer/extract_assets/image_files/YPDs/trash/test.png'
#
# image_cv2 = cv2.imread(filename=image_path)
# languages: list = ['ru']
#
# # dt_img2excel(img_path=image_path)
#
#
# print("[INFO] OCR'ing input image...")
# print(f"[INFO] OCR'ing with the following languages: {languages}")
#
# test_case: int = 1
#
# if test_case == 1:
#     # reader = easyocr.Reader(lang_list=languages, gpu=True,
#     #                         model_storage_directory='fine_tune', user_network_directory='fine_tune',
#     #                         recog_network='ru_finetune')
#     reader = easyocr.Reader(lang_list=languages, gpu=True)
#     # results = reader.readtext(image_cv2, text_threshold=0.75, contrast_ths=0.7, width_ths=1.25, height_ths=0.75,
#     #                           ycenter_ths=0.5, slope_ths=1, add_margin=0.175, decoder='wordbeamsearch', beamWidth=20,
#     #                           canvas_size=3500)
#     # Not bad
#     results = reader.readtext(image_cv2, text_threshold=0.7, contrast_ths=0.8, width_ths=1.2, height_ths=0.8,
#                               ycenter_ths=0.5, slope_ths=1, add_margin=0.200, decoder='wordbeamsearch', beamWidth=20,
#                               canvas_size=3500)
# if test_case == 2:
#     reader = easyocr.Reader(lang_list=languages, gpu=True)
#     # results = reader.readtext(image_cv2)
#     results = reader.readtext(image_cv2)
#
# # Преобразуем изображение OpenCV в формат PIL
# image_pil = Image.fromarray(cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB))
# draw = ImageDraw.Draw(image_pil)
#
# # Выберите шрифт (если нужный шрифт не установлен, вы можете указать путь к .ttf файлу)
# font = ImageFont.truetype(font="arial.ttf", size=20)  # Убедитесь, что путь к шрифту корректный
#
# for (bbox, text, prob) in results:
#     print("[INFO] {:.4f}: {}".format(prob, text))
#     # Распаковка координат
#     (tl, tr, br, bl) = bbox
#     tl = (int(tl[0]), int(tl[1]))
#     br = (int(br[0]), int(br[1]))
#
#     # Рисуем прямоугольник
#     draw.rectangle([tl, br], outline=(0, 255, 0), width=2)
#
#     # Рисуем текст
#     draw.text((tl[0], tl[1] - 10), text, fill=(0, 255, 0), font=font)
#
# # Отображаем результат
# image_pil.show()


test_img_path: str = 'pdf_appRecognizer/extract_assets/image_files/YPDs/trash/76.jpg'
test_img = cv2.imread(filename=test_img_path)
# Шаг 1 - делаем серым
gray_img = cv2.cvtColor(src=test_img, code=cv2.COLOR_BGR2GRAY)
# Шаг 2 - Уменьшаем изображение доа черных и белых пикселей (порог от белого до черного пикселя)
thresh_hold_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
# Шаг 3 - Инвертируем изображения для последующих операций
inverted_image = cv2.bitwise_not(thresh_hold_img)
# Шаг 4 - С помощью методов dilate и erode происходит утолщение всех линий. Поможет определить контуры далее.
# erode_image = cv2.erode(inverted_image, None, iterations=1)

blurred_image = cv2.GaussianBlur(inverted_image, (1, 1), 0)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))

denoised_image = cv2.morphologyEx(blurred_image, cv2.MORPH_CLOSE, kernel)
denoised_image = cv2.morphologyEx(denoised_image, cv2.MORPH_OPEN, kernel)

dilated_image = cv2.dilate(denoised_image, None, iterations=1)

cv2.imwrite(filename='temp/test.png', img=dilated_image)


# Шаг 5 - Находим все контуры из dilate_image и переносим на базовое изображение
def find_contours(dilated_img, is_debugging: bool = True):
    local_dilated_image = dilated_img
    cur_contours, cur_hierarchy = cv2.findContours(local_dilated_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if is_debugging:
        image_with_all_contours = gray_img.copy()
        cv2.drawContours(image_with_all_contours, cur_contours, -1, (0, 255, 0), 2)
        cv2.imwrite(filename='temp/test_2.png', img=image_with_all_contours)

    return cur_contours, cur_hierarchy


img_contours, _ = find_contours(dilated_img=dilated_image)


def filter_contours_and_leave_only_rectangles(contours, is_debugging: bool = True):
    rectangular_contours = []

    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        if len(approx) == 4:
            rectangular_contours.append(approx)

    if is_debugging:
        image_with_only_rectangular_contours = gray_img.copy()
        cv2.drawContours(image_with_only_rectangular_contours, rectangular_contours, -1, (0, 255, 0), 3)
        cv2.imwrite(filename='temp/test_3.png', img=image_with_only_rectangular_contours)
    return rectangular_contours


cur_rectangular_contours = filter_contours_and_leave_only_rectangles(contours=img_contours)


def crop_rectangles_to_single_image(image_path, rectangles, min_area=4000, is_debugging: bool = True):
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


dt_image = crop_rectangles_to_single_image(image_path='pdf_appRecognizer/extract_assets/image_files/YPDs/trash/76.jpg',
                                           rectangles=cur_rectangular_contours)
cv2.imwrite(filename='temp/test_5.png', img=dt_image)

# binary = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
#                                35, -5)
# Show pictures
# cv2.imshow("binary_picture", gray_img)
# cv2.imwrite(filename='temp/test.png', img=gray_img)
