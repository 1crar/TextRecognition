import os
import pytesseract
import time

from PIL import Image


class Img:
    def __init__(self, path_dir: str, exe_file: str):
        self.path_dir = path_dir
        self.exe_file = exe_file

    def get_text(self, image: str) -> str:
        # If you don't have tesseract executable in your PATH, include the following:
        pytesseract.pytesseract.tesseract_cmd: str = os.sep.join([self.path_dir, self.exe_file])
        start: float = time.time()
        extracted_text = pytesseract.image_to_string(Image.open(image))
        return f'{extracted_text}\nExtracted has been ended. \nThe time of execution is {time.time() - start} seconds'

