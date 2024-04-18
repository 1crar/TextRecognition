import os

from dotenv import load_dotenv
from classes.img import Img
from classes.text_extracter import PatternDataExtraction


load_dotenv()

path_to_tesseract: str = os.getenv('TESSERACT_PATH_DIR')
tesseract_exe: str = os.getenv('EXECUTION_FILE')

if __name__ == '__main__':
    # Извлекаем текст из пдф-файла
    image_1 = Img(path_dir=path_to_tesseract, exe_file=tesseract_exe)
    extracted_text = image_1.get_text(image=r'extract_assets/BL24-10003-1.png')

    # Извлекаем номер артикула из текста (который был извлечен из пдф)
    data_extraction = PatternDataExtraction(txt=extracted_text)
    our_data = data_extraction.extract_article_number()
    our_data = data_extraction.extract_quantity()
    print(our_data)






