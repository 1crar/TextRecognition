import json
import os

from dotenv import load_dotenv
from PIL import Image
from pdf2image import convert_from_path


load_dotenv()


class Converter:
    # Указываем путь до папки bin\ становленной программы poppler
    _poppler_path: str = os.getenv('POPPLER_PATH')

    def __init__(self):
        """
        Скорее всего, также нужно будет указать путь до исполнительного файла, как и в классе Img
        def __init__(self, path_dir: str, exe_file: str):
            self.path_dir = path_dir
            self.exe_file = exe_file
        """

        pass

    @staticmethod
    def from_pdf_to_png(pdf_path: str, to_save: str) -> None:
        # Второй параметр влияет на качество изображения (при необходимости можно поставить выше значение)
        pages = convert_from_path(pdf_path=pdf_path, dpi=200, poppler_path=Converter._poppler_path)
        # Сохраняем каждое изображение в PNG формате
        for i, page in enumerate(pages):
            page.save(f'{to_save}_{i + 1}.png', 'PNG')

    @staticmethod
    def from_png_to_pdf(png_path: str, to_save: str) -> None:
        pdf_file = Image.open(png_path)
        pdf_file.convert('RGB')
        # Сохраняем конфертированный из png в pdf файл
        pdf_file.save(fp=to_save, quality=100)


def from_txt_to_json(input_txt_path: str, output_json_path: str) -> None:
    with open(file=input_txt_path, mode='r', encoding='utf-8') as txt_file:
        lines: list = [line.rstrip() for line in txt_file]
        cur_dict: dict = {}

        for line in lines:
            key, value = line.split("`")
            cur_dict[key[2:]] = value       # Игнорируем первые два индекса ключа

    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(cur_dict, f, ensure_ascii=False, indent=3)


from_txt_to_json(input_txt_path=r'C:\Users\serge\Рабочий стол\all_labels\6.txt',
                 output_json_path=r'C:\Users\serge\Рабочий стол\all_labels\6.json')
