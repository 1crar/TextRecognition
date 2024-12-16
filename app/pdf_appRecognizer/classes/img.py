from typing import Any, Sequence

import copy
import cv2
import easyocr
import os
import pytesseract
import time
import torch
import subprocess

import numpy as np

from dotenv import load_dotenv
from img2table.ocr import EasyOCR
from img2table.document import Image as ImageDoc
from PIL import Image as Img
from PIL import ImageEnhance
from numpy import ndarray
from super_image import DrlnModel

load_dotenv()
TESSERACT_OCR: str = os.getenv('TESSERACT')


class ImageTest:
    "Old version of image extraction"

    def __init__(self, _path_dir: str, exe_file: str):
        self._path_dir = _path_dir
        self.exe_file = exe_file

    def get_text(self, image: str) -> str:
        # If you don't have tesseract executable in your PATH, include the following:
        pytesseract.pytesseract.tesseract_cmd: str = os.sep.join([self._path_dir, self.exe_file])
        start: float = time.time()
        extracted_text = pytesseract.image_to_string(Img.open(image))
        return f'{extracted_text}\nExtracted has been ended. \nThe time of execution is {time.time() - start} seconds'


class ImageDataExtracter:
    """
    Current class for extraction data from image
    """

    def __init__(self, path_dir: str, image_file: str, path_to_tesseract: str, language: str):
        self._path_dir = path_dir
        self._image_file = image_file
        self._full_path = f'{self._path_dir}/{self._image_file}'
        self._language = language

        self._path_to_tesseract = path_to_tesseract

    @property
    def image_path(self) -> str:
        return self._full_path

    @property
    def tesseract_path(self) -> str:
        return self._path_to_tesseract

    @property
    def language(self) -> str:
        return self._language

    def extract_data_from_image(self) -> str:
        image = cv2.imread(self._full_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Применяем пороговое преобразование
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

        # If you don't have tesseract executable in your PATH, include the following:
        pytesseract.pytesseract.tesseract_cmd: str = self._path_to_tesseract

        text = pytesseract.image_to_string(thresh, lang=self._language)
        return text

    def tesseract_extraction(self) -> str:
        img = Img.open(self._full_path)

        pytesseract.pytesseract.tesseract_cmd: str = self._path_to_tesseract

        text = pytesseract.image_to_string(img, lang=self.language)

        data = [line.split() for line in text.splitlines() if line.strip()]

        # df = pd.DataFrame(data)
        # print(df)

        return text[:-1]


class ImageTableExtracter:
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
            cv2.imwrite(filename='../../temp/test.png', img=self.denoised_image)

        # return self.denoised_image

    def find_contours(self, is_debugging: bool = True):
        contours, _ = cv2.findContours(self.denoised_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if is_debugging:
            image_with_all_contours = self.gray_img.copy()
            cv2.drawContours(image_with_all_contours, contours, -1, (0, 255, 0), 2)
            cv2.imwrite(filename='../../temp/test_2.png', img=image_with_all_contours)

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
            cv2.imwrite(filename='../../temp/test_3.png', img=image_with_only_rectangular_contours)

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
            cv2.imwrite(filename='../../temp/test_combined.png', img=debug_image)

        # cv2.imwrite(filename='../../temp/test_4_final_result.png', img=cropped_image)
        cv2.imwrite(filename=dt_coloured_img, img=cropped_image)
        return cropped_image


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
    # img_enhanced.show() - для отладки
    # img_enhanced.show()
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


# Get the list of available languages
def tesseract_languages(path_to_tesseract: str) -> list[str]:
    # If you don't have tesseract executable in your PATH, include the following:
    pytesseract.pytesseract.tesseract_cmd: str = path_to_tesseract
    languages: list[str] = pytesseract.get_languages()
    return languages


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


def image_processing(coloured_image_file: str, uncoloured_image_file: str) -> np.ndarray:
    folder_path: str = '../extract_assets/image_files/YPDs/trash/'

    img_instance = ImageTableExtracter(image_path=f'{folder_path}/{uncoloured_image_file}')

    img_instance.image_processing()
    img_instance.find_contours()

    img_instance.filter_contours_and_leave_only_rectangles(index=0.01)
    # возвращаем не путь до обрезанного изображения, а np.array
    dt_im = img_instance.crop_rectangles_to_single_image(dt_coloured_img=f'{folder_path}{coloured_image_file}')

    return dt_im


def cropped_to_dt_img_files(folder_path: str):
    # dt_data = None
    for img_file in os.listdir(path=folder_path):
        if img_file.endswith('.png'):
            # Создаем имя для обрезанного изображения
            output_file_name = f'{img_file.replace('.png', '')}_cropped.png'
            # Обрезаем изображение
            cropped_based_img(input_file=f'{folder_path}{img_file}', output_file=f'{folder_path}{output_file_name}')
            # Создаем имя для серого изображения
            uncoloured_img_file: str = output_file_name.replace('.png', '_uncoloured.png')
            # На основе обрезанного изображения мы улучшаем его качество и делаем его серым
            improve_img_quality(img_path=f'{folder_path}/{output_file_name}',
                                output_path=f'{folder_path}/{uncoloured_img_file}',
                                sharpness=14, contrast=3, blur=1)
            # Извлекаем данные из табличной части
            dt_data = image_processing(coloured_image_file=output_file_name,
                                       uncoloured_image_file=uncoloured_img_file)

        if img_file.endswith('.jpg'):
            output_file_name = f'{img_file.replace('.jpg', '')}_cropped.jpg'
            cropped_based_img(input_file=f'{folder_path}{img_file}', output_file=f'{folder_path}{output_file_name}')

            uncoloured_img_file: str = output_file_name.replace('.jpg', '_uncoloured.jpg')

            improve_img_quality(img_path=f'{folder_path}/{output_file_name}',
                                output_path=f'{folder_path}/{uncoloured_img_file}',
                                sharpness=14, contrast=3, blur=1)

            dt_data = image_processing(coloured_image_file=output_file_name,
                                       uncoloured_image_file=uncoloured_img_file)

    # return dt_data


def removed_uncoloured_images(folder_path: str):
    for img_file in os.listdir(path=folder_path):
        if 'uncoloured' in img_file:
            os.remove(f'{folder_path}{img_file}')


def img_app_run():              # is_already_cropped: bool = True
    folder_path: str = '../extract_assets/image_files/YPDs/trash/'

    cropped_to_dt_img_files(folder_path=folder_path)
    removed_uncoloured_images(folder_path=folder_path)

    # coloured_img_file: str = '24_cropped.png'
    # uncoloured_img_file: str = '24_uncoloured_cropped.png'
    #
    # improve_img_quality(img_path=f'{folder_path}/{coloured_img_file}',
    #                     output_path=f'{folder_path}/{uncoloured_img_file}',
    #                     sharpness=14, contrast=3, blur=1)
    # image_processing(coloured_image_file=coloured_img_file, uncoloured_image_file=uncoloured_img_file)
    # img_instance = ImageTableExtracter(image_path=f'{folder_path}/{uncoloured_img_file}')
    #
    # img_instance.image_processing()
    # img_instance.find_contours()
    #
    # img_instance.filter_contours_and_leave_only_rectangles(index=0.01)
    # # возвращаем не путь до обрезанного изображения, а np.array
    # dt_im = img_instance.crop_rectangles_to_single_image(dt_coloured_img=f'{folder_path}{coloured_img_file}')
    #
    # # print(dt_im, sep='\n')


def erode_vertical_lines(inverted_image: np.ndarray) -> np.ndarray:
    hor = np.array([[1, 1, 1, 1, 1, 1]])

    vertical_lines_eroded_image = cv2.erode(inverted_image, hor, iterations=10)
    vertical_lines_eroded_image = cv2.dilate(vertical_lines_eroded_image, hor, iterations=10)

    return vertical_lines_eroded_image


def erode_horizontal_lines(inverted_image: np.ndarray) -> np.ndarray:
    ver = np.array([[1],
                    [1],
                    [1],
                    [1],
                    [1],
                    [1],
                    [1]])
    horizontal_lines_eroded_image = cv2.erode(inverted_image, ver, iterations=10)
    horizontal_lines_eroded_image = cv2.dilate(horizontal_lines_eroded_image, ver, iterations=10)

    return horizontal_lines_eroded_image


def combine_eroded_images(vertical_lines_eroded_image: np.ndarray,
                          horizontal_lines_eroded_image: np.ndarray) -> np.ndarray:
    combined_image = cv2.add(vertical_lines_eroded_image, horizontal_lines_eroded_image)
    return combined_image


def dilate_lines_thicker(combined_image: np.ndarray) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    combined_image_dilated = cv2.dilate(combined_image, kernel, iterations=3)

    return combined_image_dilated


def remove_noise_with_erode_and_dilate(image_without_lines: np.ndarray) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    image_without_lines_noise_removed = cv2.erode(image_without_lines, kernel, iterations=1)
    image_without_lines_noise_removed = cv2.dilate(image_without_lines_noise_removed, kernel, iterations=1)

    return image_without_lines_noise_removed


def dilate_image(denoised_image: np.ndarray) -> np.ndarray:
    kernel_to_remove_gaps_between_words = np.array([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    ])
    dilated_image = cv2.dilate(denoised_image, kernel_to_remove_gaps_between_words, iterations=5)
    simple_kernel = np.ones((5, 5), np.uint8)

    denoised_dilated_image = cv2.dilate(dilated_image, simple_kernel, iterations=2)
    return denoised_dilated_image


def find_word_contours(denoised_dilated_image: np.ndarray) -> Sequence[ndarray | Any]:
    result = cv2.findContours(denoised_dilated_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = result[0]

    return contours


def convert_contours_to_bounding_boxes(contours: Sequence[ndarray | Any],
                                       original_img: np.ndarray) -> Sequence[ndarray | Any]:
    bounding_boxes = []
    image_with_all_bounding_boxes = original_img.copy()

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        bounding_boxes.append((x, y, w, h))

        # Строки ниже нужны для отладки
        image_with_all_bounding_boxes = cv2.rectangle(image_with_all_bounding_boxes,
                                                      (x, y), (x + w, y + h), (0, 255, 0), 1)
        cv2.imwrite(filename='../../temp/test_14_bounding_boxes.png', img=image_with_all_bounding_boxes)

    return bounding_boxes


def get_mean_height_of_bounding_boxes(bounding_boxes: Sequence[ndarray | Any]) -> np.floating:
    heights = []

    for bounding_box in bounding_boxes:
        x, y, w, h = bounding_box
        heights.append(h)

    return np.mean(heights)


def sort_bounding_boxes_by_y_coordinate(bounding_boxes: Sequence[ndarray | Any]) -> Sequence[ndarray | Any]:
    bounding_boxes = sorted(bounding_boxes, key=lambda x: x[1])
    return bounding_boxes


def club_all_bounding_boxes_by_similar_y_coordinates_into_rows(mean_height: np.floating,
                                                               sorted_bounding_boxes: Sequence[ndarray | Any]) ->\
        Sequence[ndarray | Any]:

    rows = []
    half_of_mean_height = mean_height / 2
    current_row = [sorted_bounding_boxes[0]]

    for bounding_box in sorted_bounding_boxes[1:]:
        current_bounding_box_y = bounding_box[1]
        previous_bounding_box_y = current_row[-1][1]
        distance_between_bounding_boxes = abs(current_bounding_box_y - previous_bounding_box_y)

        if distance_between_bounding_boxes <= half_of_mean_height:
            current_row.append(bounding_box)
        else:
            rows.append(current_row)
            current_row = [bounding_box]

    rows.append(current_row)
    return rows


def sort_all_rows_by_x_coordinate(rows: Sequence[ndarray | Any]):
    for row in rows:
        row.sort(key=lambda x: x[0])
    return rows


def get_result_from_tesseract(image_path: str) -> str:
    output = subprocess.getoutput('tesseract ' + image_path + ' - -l rus --oem 3 --psm 7 --dpi 72 -c tessedit_char_whitelist=tessedit_char_whitelist="АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ0123456789().calmg* "')
    output = output.strip()

    # print(f'The type of variable output is {type(output)}')
    return output


def get_result_from_easyocr(image_path: str) -> list | str | None:
    extracted_data: list[str] = []
    img_cv2 = cv2.imread(filename=image_path)

    reader = easyocr.Reader(lang_list=['ru'], gpu=True)
    output = reader.readtext(image=img_cv2)

    for (_, text, _) in output:
        # print(f"[INFO]: {text}")
        extracted_data.append(text)

    if len(extracted_data) == 1:
        return extracted_data[0]

    if len(extracted_data) == 0:    # Бывает и пустой список
        return None

    return extracted_data


def crop_each_bounding_box_and_ocr(rows: Sequence[ndarray | Any], based_image: np.ndarray) -> list:
    table = []
    current_row = []
    image_number = 0

    for row in rows:
        for bounding_box in row:
            x, y, w, h = bounding_box
            y = y - 5

            cropped_image = based_image[y:y+h, x:x+w]
            image_slice_path: str = f'../../temp/ocr_slices/img_{image_number}.png'
            try:
                cv2.imwrite(image_slice_path, cropped_image)
                results_from_ocr = get_result_from_easyocr(image_path=image_slice_path)

                if results_from_ocr is None:
                    continue
                else:
                    current_row.append(results_from_ocr)

                image_number += 1
            except cv2.error as e:
                print(f"Ошибка во время записи {e}")

        table.append(current_row)
        current_row = []

    return table


def generate_csv_file(table: list | str, csv_filename: str):
    with open(csv_filename, "w", encoding='utf-8') as f:
        for row in table:
            if isinstance(row, list):
                # Если строка у нас список
                f.write(",".join(str(item) for item in row) + "\n")
            else:
                f.write(row + "\n")


def recover_image(inverted_image: np.ndarray) -> np.ndarray:
    recovered_img = cv2.bitwise_not(src=inverted_image)
    recovered_img = cv2.threshold(recovered_img, 127, 255, cv2.THRESH_BINARY)[1]

    return recovered_img


def detect_datatable_part(ypd_img_path: str, output_filename: str, offset: int = 0,
                          is_erode: bool = True, is_debugging: bool = True) -> str:
    ocr_detector = EasyOCR(lang=['ru'])

    # Создаем экземпляр класса документ
    doc_dt = ImageDoc(src=ypd_img_path)
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
        # Динамически будет создаваться отступ в пропорции 3% от всей высоты таблицы
        dynamic_lower_offset = int(table_height * 0.03)

        # Создаем общий отступ
        total_y_min = max(0, min_y1 - offset)
        total_y_max = max_y2 + offset + dynamic_lower_offset

        # Обрезаем всю табличную часть с учетом границ и отступа
        crop_dt_img = table_img[
                      total_y_min:total_y_max,
                      max(0, min_x1 - offset):max_x2 + offset
                  ]
        if is_debugging:
            # Следующие строки нужны для отладки процесса распознавания табличной части
            detected_lines = copy.deepcopy(table_img)
            # Прорисовка границ табличной части
            for row in table.content.values():
                for cell in row:
                    x1, y1, x2, y2 = cell.bbox.x1, cell.bbox.y1, cell.bbox.x2, cell.bbox.y2
                    cv2.rectangle(detected_lines, (x1, y1), (x2, y2), (0, 0, 255), 2)
            # Сохраняем изображение табличной части
            # cv2.imwrite(filename=temp_filename, img=detected_lines)

        try:
            if is_erode:
                crop_dt_img = cv2.erode(crop_dt_img, kernel=np.ones((2, 2)), iterations=1)
            cv2.imwrite(output_path, crop_dt_img)
        except Exception as e:
            # print(f"ERROR: {e}")
            raise Exception(f"ERROR: ошибка записи файла - {e}")

    return output_path


def test_2():
    cur_img_path: str = '../../temp/test_4_final_result_x2.png'
    cur_img: np.ndarray = cv2.imread(filename=cur_img_path)

    gray_img = cv2.cvtColor(src=cur_img, code=cv2.COLOR_BGR2GRAY)
    threshold_img = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)[1]
    inverted_img = cv2.bitwise_not(src=threshold_img)

    cv2.imwrite(filename='../../temp/test_5_inverted_dt.png', img=inverted_img)

    v_lines_erode_image = erode_vertical_lines(inverted_image=inverted_img)
    cv2.imwrite(filename='../../temp/test_6_v_eroded_dt.png', img=v_lines_erode_image)

    h_lines_erode_image = erode_horizontal_lines(inverted_image=inverted_img)
    cv2.imwrite(filename='../../temp/test_7_h_eroded_dt.png', img=h_lines_erode_image)

    combined_lines_image = combine_eroded_images(vertical_lines_eroded_image=v_lines_erode_image,
                                                 horizontal_lines_eroded_image=h_lines_erode_image)
    cv2.imwrite(filename='../../temp/test_8_combined_lines_dt.png', img=combined_lines_image)

    dilated_combined_image = dilate_lines_thicker(combined_image=combined_lines_image)
    cv2.imwrite(filename='../../temp/test_9_combined_lines_dilated_dt.png', img=dilated_combined_image)

    # Убираем линии
    image_without_lines = cv2.subtract(inverted_img, dilated_combined_image)
    cv2.imwrite(filename='../../temp/test_10_without_lines_dt.png', img=image_without_lines)

    # Перед denoised мы можем попробовать заапскейлить изображение!!!

    denoised_image = remove_noise_with_erode_and_dilate(image_without_lines=image_without_lines)
    cv2.imwrite(filename='../../temp/test_11_denoised_dt.png', img=denoised_image)

    # recovered_image = recover_image(inverted_image=denoised_image)
    # cv2.imwrite(filename='../../temp/test_13_recovered_dt.png', img=recovered_image)

    denoised_dilated_image = dilate_image(denoised_image=denoised_image)
    cv2.imwrite(filename='../../temp/test_12_denoised_dilated_dt.png', img=denoised_dilated_image)

    word_contours = find_word_contours(denoised_dilated_image=denoised_dilated_image)
    cv2.drawContours(cur_img, word_contours, -1, (0, 255, 0), 3)

    cv2.imwrite(filename='../../temp/test_13_contoured_dt.png', img=cur_img)

    bounding_boxes = convert_contours_to_bounding_boxes(contours=word_contours, original_img=cur_img)
    mean_height_bounding_boxes: np.floating = get_mean_height_of_bounding_boxes(bounding_boxes=bounding_boxes)

    print(f'Средняя высота bounding boxes - {mean_height_bounding_boxes}')

    sorted_bounding_boxes = sort_bounding_boxes_by_y_coordinate(bounding_boxes=bounding_boxes)

    print(f'bounding_boxes успешно отсортированы по y координате')

    dt_rows = club_all_bounding_boxes_by_similar_y_coordinates_into_rows(sorted_bounding_boxes=sorted_bounding_boxes,
                                                                         mean_height=mean_height_bounding_boxes)
    sorted_dt_rows = sort_all_rows_by_x_coordinate(rows=dt_rows)
    print(f'Полученные строки отсортированы - sorted_dt_rows')

    table = crop_each_bounding_box_and_ocr(rows=sorted_dt_rows, based_image=cur_img)
    print(f'table извлечена через EasyOCR')
    generate_csv_file(table=table, csv_filename='../../temp/csv_files/test_3.csv')
    print('Извлеченные данные из table записаны в csv файл')


# test_2()


def test_3():
    cur_path: str = f'../extract_assets/image_files/YPDs/trash/'
    cur_model = DrlnModel.from_pretrained('eugenesiow/drln', scale=4)

    for dt_cropped_img in os.listdir(path=cur_path):
        if 'cropped' in dt_cropped_img:
            print(dt_cropped_img[:-4])
            # Скейлим изображение
            dt_upscale_img = upscale_image(path_to_based_img=f'{cur_path}{dt_cropped_img}',
                                           path_to_upscaled_img=f'{cur_path}{dt_cropped_img}',
                                           model=cur_model)
            #
            cur_img: np.ndarray = cv2.imread(filename=dt_upscale_img)

            reader = easyocr.Reader(lang_list=['ru'], gpu=True)
            output = reader.readtext(image=cur_img)

            data: list = []
            for (_, text, _) in output:
                data.append(text)
            generate_csv_file(table=data, csv_filename=f'../../temp/csv_files/{dt_cropped_img[:-4]}.csv')


# test_3()


def test_4():
    start = time.time()

    cur_path: str = f'../extract_assets/image_files/YPDs/'

    ignored_folders: set = {'sample', 'trash', 'dt_part_not_trash_YPDs'}
    counter_files: int = 0

    for img_file in os.listdir(path=cur_path):
        try:
            if img_file not in ignored_folders:
                dt_img = detect_datatable_part(ypd_img_path=f'{cur_path}{img_file}',
                                               output_filename=f'{cur_path}dt_part_not_trash_YPDs/{img_file[:-4]}_cropped.'
                                                               f'{img_file[-3:]}',
                                               is_debugging=False)
                print(f'INFO: Табличная часть упд успешно записана {dt_img}')
                counter_files += 1
        except Exception as e:
            raise ValueError(f"Не удалось записать изображение {e}")

    print(f'INFO: Кол-во обработанных изображений в директории {cur_path} --- {counter_files}')
    end = time.time() - start
    print(f'Время выполнения команды: {end:10.2f}')


test_4()
