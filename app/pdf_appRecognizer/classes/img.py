import cv2
import os
import pytesseract
import time

import pandas as pd

from PIL import Image


class Img:
    "Old version of image extraction"
    def __init__(self, _path_dir: str, exe_file: str):
        self._path_dir = _path_dir
        self.exe_file = exe_file

    def get_text(self, image: str) -> str:
        # If you don't have tesseract executable in your PATH, include the following:
        pytesseract.pytesseract.tesseract_cmd: str = os.sep.join([self._path_dir, self.exe_file])
        start: float = time.time()
        extracted_text = pytesseract.image_to_string(Image.open(image))
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


# Get the list of available languages
def tesseract_languages(path_to_tesseract: str) -> list[str]:
    # If you don't have tesseract executable in your PATH, include the following:
    pytesseract.pytesseract.tesseract_cmd: str = path_to_tesseract
    languages: list[str] = pytesseract.get_languages()
    return languages

