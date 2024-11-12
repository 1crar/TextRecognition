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
        img = Image.open(self._full_path)

        pytesseract.pytesseract.tesseract_cmd: str = self._path_to_tesseract

        text = pytesseract.image_to_string(img, lang=self.language)

        data = [line.split() for line in text.splitlines() if line.strip()]

        # df = pd.DataFrame(data)
        # print(df)

        return text[:-1]


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

    # Если есть видеокарта, то использует ее вместо процессора
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


# Get the list of available languages
def tesseract_languages(path_to_tesseract: str) -> list[str]:
    # If you don't have tesseract executable in your PATH, include the following:
    pytesseract.pytesseract.tesseract_cmd: str = path_to_tesseract
    languages: list[str] = pytesseract.get_languages()
    return languages

