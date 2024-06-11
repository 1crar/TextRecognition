import os
import time
import camelot
import csv
import re

from dotenv import load_dotenv
from classes.pdf import PdfAnalyzer, PdfTabula, PdfCamelot, PdfTextReader, extract_tables
from classes.img import ImageDataExtracter, tesseract_languages
from classes.text_extracter import InnInvoiceDataExtraction, DictToJson

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
        print(result, '---------------Execution time---------------', f'{(time.time() - start):.2f}', sep='\n')

        # записываем результат в extracted_results/Image_result.txt
        with open(file='extracted_results/Image_result.txt', mode='w', encoding='utf-8') as f:
            print(result, file=f)
    else:
        start = time.time()
        # Camelot extraction (from pdf(dataTable) to csv)
        camelot_instance = PdfCamelot(path_dir='extract_assets/input_files/for_camelot_extraction',
                                      pdf_file='УПД №1196036_0038 от 04.06.24.pdf')

        tables = camelot_instance.read_tables()
        camelot_instance.write_to_csv(tables=tables, file_csv_name='Camelot_result.csv')

        # Re extraction (inn/kpp and invoice) to json
        pdf = PdfTextReader(path_dir='extract_assets/input_files/for_camelot_extraction',
                            pdf_file='УПД №1196036_0038 от 04.06.24.pdf')
        text = pdf.extract_text_from_pdf()

        my_regulars: 'InnInvoiceDataExtraction' = InnInvoiceDataExtraction(text=text)

        inn_kpp_lst: list = my_regulars.inn_and_kpp_extract()
        invoices: list = my_regulars.invoice_extract()

        data = my_regulars.data_collect(inn_kpp_nums=inn_kpp_lst, invoices=invoices)
        DictToJson.write_to_json(collection=data)

        print(data, '---------------Execution time---------------', f'{(time.time() - start):.2f}', sep='\n')
