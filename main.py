import os
import time
import camelot
import csv
import re

from dotenv import load_dotenv
from classes.pdf import PdfAnalyzer, PdfTabula, PdfCamelot, PdfTextReader, extract_tables
from classes.img import ImageDataExtracter, tesseract_languages
from classes.text_extracter import InnInvoiceDataExtraction

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
        # camelot_instance = PdfCamelot(path_dir='extract_assets/input_files/for_camelot_extraction',
        #                               pdf_file='УПД №1196036_0038 от 04.06.24.pdf')
        #
        # tables = camelot_instance.read_tables()
        # camelot_instance.write_to_csv(tables=tables, file_csv_name='Camelot_result.csv')

        pdf = PdfTextReader(path_dir='extract_assets/input_files/for_camelot_extraction',
                            pdf_file='УПД №1196036_0038 от 04.06.24.pdf')
        text = pdf.extract_text_from_pdf()

        my_regulars = InnInvoiceDataExtraction(text=text)
        print(my_regulars.inn_and_kpp_extract())


        # # Регулярное выражение для извлечения значений ИНН/КПП
        # pattern_inn_kpp = re.compile(r'ИНН/КПП (\d{10}) / (\d{9})')
        #
        # # Регулярное выражение для извлечения номера Счет-фактуры
        # pattern_invoice_number = re.compile(r'УПД Счет-фактура № (\d{7}/\d{4}) от (\d{2}.\d{2}.\d{4})')
        #
        # # Поиск значений ИНН/КПП и номера Счет-фактуры в тексте
        # inn_kpp_matches = pattern_inn_kpp.findall(text)
        # invoice_number_matches = pattern_invoice_number.findall(text)
        #
        # # Вывод результатов
        # for matches in inn_kpp_matches:
        #     print(f"ИНН: {matches[0]}, КПП: {matches[1]}")
        #
        # for matches in invoice_number_matches:
        #     print(f"Номер Счет-фактуры: {matches[0]}")
        #
        # with open(file='extracted_results/PyPdf_result.txt', mode='w', encoding='utf-8') as file:
        #     file.write(text)

