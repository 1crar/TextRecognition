import os
import time

from dotenv import load_dotenv
from classes.pdf import PdfAnalyzer, extract_tables
from classes.img import ImageDataExtracter, tesseract_languages

# Загружаем переменные из файла .env
load_dotenv()

path_to_tesseract: str = os.getenv('TESSERACT_PATH_DIR')
tesseract_exe: str = os.getenv('EXECUTION_FILE')

# Позволяет выбрать из какого формата будем извлекать (если True, то из картинки. False - из пдф)
IS_IMAGE: bool = True

if __name__ == '__main__':
    # Write list of available languages to file
    with open(file='tesseract_languages.txt', mode='w', encoding='utf-8') as f:
        print(tesseract_languages(path_to_tesseract=f'{path_to_tesseract}/{tesseract_exe}'), file=f)

    start = time.time()
    if IS_IMAGE:
        image = ImageDataExtracter(path_dir='extract_assets/input_files', image_file='img_1.png',
                                   path_to_tesseract=f'{path_to_tesseract}/{tesseract_exe}', language='rus')
        result = image.extract_data_from_image()
        print(result, '---------------Execution time---------------', (time.time() - start), sep='\n')

        # записываем результат в extracted_results/Image_result.txt
        with open(file='extracted_results/Image_result.txt', mode='w', encoding='utf-8') as f:
            print(result, file=f)
    else:
        # Создаем экземпляр класса и присваиваем конкретный pdf-документ, который будем парсить
        pdf_example = PdfAnalyzer(path_dir='extract_assets', pdf_file='pdf_1.pdf')
        print(pdf_example.full_path)

        # Условие срабатывает, если в pdf-документе есть табличная часть
        if pdf_example.is_contains_data_table():
            # Функция возвращает данные в виде списка, извлеченные из табличной части
            pdf_table = extract_tables(path_to_pdf=pdf_example.full_path)
            print(f'данные в pdf_table\n{pdf_table}')