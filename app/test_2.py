import cv2
import copy
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


image_path: str = f'pdf_appRecognizer/extract_assets/image_files/YPDs/test_2/dt_cropped/10_restored_PD_1_without_lines_dt.png'
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
test_case: int = 1
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
    results = reader.readtext(image_cv2, text_threshold=0.0, contrast_ths=0.8, width_ths=0.9, height_ths=0.7,
                              ycenter_ths=0.5, slope_ths=1, add_margin=0.2, decoder='wordbeamsearch', beamWidth=20,
                              canvas_size=3500, y_ths=0.8)
if test_case == 2:
    reader = easyocr.Reader(lang_list=languages, gpu=True)
    # results = reader.readtext(image_cv2)
    results = reader.readtext(image_cv2)

# print(results)
#
# # Преобразуем изображение OpenCV в формат PIL
image_pil = Image.fromarray(cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB))
draw = ImageDraw.Draw(image_pil)
#
# # Выберите шрифт (если нужный шрифт не установлен, вы можете указать путь к .ttf файлу)
font = ImageFont.truetype(font="arial.ttf", size=20)  # Убедитесь, что путь к шрифту корректный
#
data_table: list = []
detected_bboxes = []


def is_intersecting(bbox1, bbox2):
    # Получаем координаты
    (tl1, tr1, br1, bl1) = bbox1
    (tl2, tr2, br2, bl2) = bbox2

    # Проверяем пересечение
    if tl1[0] < br2[0] and br1[0] > tl2[0] and tl1[1] < br2[1] and br1[1] > tl2[1]:
        return True
    return False


filtered_results = []
used_indices = set()

# merged_bboxes = merge_bboxes(bboxes=detected_bboxes, delta_x=0.1, delta_y=3.0)

for i in range(len(results)):
    if i in used_indices:
        continue

    bbox1, text1, prob1 = results[i]
    filtered_results.append(results[i])

    for j in range(i + 1, len(results)):
        if j in used_indices:
            continue

        bbox2, text2, prob2 = results[j]

        if is_intersecting(bbox1, bbox2):
            filtered_results.append(results[j])
            used_indices.add(j)


for (bbox, text, prob) in filtered_results:
    print("[INFO] {:.4f}: {}".format(prob, text))
    # Распаковка координат
    (tl, tr, br, bl) = bbox
    tl = (int(tl[0]), int(tl[1]))
    br = (int(br[0]), int(br[1]))

    # Рисуем прямоугольник
    draw.rectangle([tl, br], outline=(0, 255, 0), width=2)

    # Рисуем текст
    draw.text((tl[0], tl[1] - 10), text, fill=(0, 255, 0), font=font)
    data_table.append(text)

from pdf_appRecognizer.classes.img import generate_csv_file

generate_csv_file(table=data_table,
                  csv_filename='pdf_appRecognizer/extract_assets/image_files/YPDs/test_2/csv_files/YPD_1_ver3.csv')

# Отображаем результат
image_pil.show()


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