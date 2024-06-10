import os
import time
import camelot
import csv

from dotenv import load_dotenv
from classes.pdf import PdfAnalyzer, PdfTabula, PdfCamelot, extract_tables
from classes.img import ImageDataExtracter, tesseract_languages

# Загружаем переменные из файла .env
load_dotenv()

path_to_tesseract: str = os.getenv('TESSERACT_PATH_DIR')
tesseract_exe: str = os.getenv('EXECUTION_FILE')

# Позволяет выбрать из какого формата будем извлекать (если True, то из картинки. False - из пдф)
IS_IMAGE: bool = False

if __name__ == '__main__':
    # Write list of available languages to file
    with open(file='tesseract_languages.txt', mode='w', encoding='utf-8') as f:
        print(tesseract_languages(path_to_tesseract=f'{path_to_tesseract}/{tesseract_exe}'), file=f)

    start = time.time()
    if IS_IMAGE:
        image = ImageDataExtracter(path_dir='extract_assets/input_files', image_file='IMG_20240603_191413.jpg',
                                   path_to_tesseract=f'{path_to_tesseract}/{tesseract_exe}', language='rus')
        result = image.extract_data_from_image()
        print(result, '---------------Execution time---------------', (time.time() - start), sep='\n')

        # записываем результат в extracted_results/Image_result.txt
        with open(file='extracted_results/Image_result.txt', mode='w', encoding='utf-8') as f:
            print(result, file=f)
    else:
        tables = camelot.read_pdf(
            filepath='extract_assets/input_files/for_camelot_extraction/УПД №1196036_0038 от 04.06.24.pdf',
            pages='all')

        print(len(tables))

        with open(file='extracted_results/Camelot_result.csv', mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)

            for table in tables:

                cleaned_table = []
                for row in table.data[1:]:
                    cleaned_row = [item.replace('\n', ' ') for item in row]
                    cleaned_table.append(cleaned_row)
                print(cleaned_table)

                for row in cleaned_table:
                    writer.writerow(row)


        # for table in tables:
        #     cleaned_table: list[list] = []
        #
        #     for row in table.data[1:]:
        #         # Очищаем от переходов
        #         cleaned_row = [item.replace('\n', ' ') for item in row]
        #         cleaned_table.append(cleaned_row)
        #
        #     print(cleaned_table)
        #
        #     with open(file='extracted_results/Camelot_result.csv', mode='w', newline='', encoding='utf-8') as file:
        #         writer = csv.writer(file)
        #
        #         for row in cleaned_table:
        #             writer.writerow(row)
