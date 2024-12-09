import cv2
import os
import pytesseract
import time
import torch

import numpy as np
import pandas as pd

from dotenv import load_dotenv
from PIL import Image as Img
from PIL import ImageEnhance


load_dotenv()
TESSERACT_OCR: str = os.getenv('TESSERACT')


class Image:
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

    def crop_rectangles_to_single_image(self, min_area=4000, is_debugging: bool = True):
        img = cv2.imread(filename=self.image_path)

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

        cv2.imwrite(filename='../../temp/test_4.png', img=cropped_image)
        # return cropped_image


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


def cropped_img_files(folder_path: str):
    for img_file in os.listdir(path=folder_path):
        output_file_name: str = ''

        if img_file.endswith('.png'):
            output_file_name = f'{img_file.replace('.png', '')}_cropped.png'
        if img_file.endswith('.jpg'):
            output_file_name = f'{img_file.replace('.jpg', '')}_cropped.jpg'

        cropped_based_img(input_file=f'{folder_path}{img_file}', output_file=f'{folder_path}{output_file_name}')


def test(is_already_cropped: bool = True):
    folder_path: str = '../extract_assets/image_files/YPDs/trash/'

    if is_already_cropped:
        cropped_img_files(folder_path=folder_path)

    img_filename: str = '24_cropped.png'

    improve_img_quality(img_path=f'{folder_path}/{img_filename}', output_path=f'{folder_path}/{img_filename}',
                        sharpness=14, contrast=3, blur=1)

    img_instance = ImageTableExtracter(image_path=f'{folder_path}/{img_filename}')

    img_instance.image_processing()
    img_instance.find_contours()

    img_instance.filter_contours_and_leave_only_rectangles(index=0.01)
    img_instance.crop_rectangles_to_single_image()


test(is_already_cropped=False)

