import os
import time
import logging

from dotenv import load_dotenv
from log_config.log_config import activate_logging

from classes.data_json import DataCollection, DataCleaning
from classes.pdf import PdfAnalyzer, PdfTabula, PdfCamelot, PdfTextReader, extract_tables
from classes.img import ImageDataExtracter, tesseract_languages
from classes.text_extracter import InnInvoiceDataExtraction, DictToJson

# Загружаем переменные из файла .env
load_dotenv()
# Подключаем главный лог
logger = logging.getLogger()

path_to_tesseract: str = os.getenv('TESSERACT_PATH_DIR')
tesseract_exe: str = os.getenv('EXECUTION_FILE')

# Позволяет выбрать из какого формата будем извлекать (если True, то из картинки. False - из пдф)
IS_IMAGE: bool = False

if __name__ == '__main__':
    # Write list of available languages to file
    with open(file='tesseract_languages.txt', mode='w', encoding='utf-8') as f:
        print(tesseract_languages(path_to_tesseract=f'{path_to_tesseract}/{tesseract_exe}'), file=f)

    if IS_IMAGE:
        activate_logging()
        start = time.time()
        image = ImageDataExtracter(path_dir='extract_assets/input_files', image_file='IMG_20240603_191413.jpg',
                                   path_to_tesseract=f'{path_to_tesseract}/{tesseract_exe}', language='rus')
        result = image.extract_data_from_image()
        print(result, '---------------Execution time---------------', f'{(time.time() - start):.2f}', sep='\n')

        # записываем результат в extracted_results/Image_result.txt
        with open(file='extracted_results/Image_result.txt', mode='w', encoding='utf-8') as f:
            print(result, file=f)
    else:
        start = time.time()
        activate_logging()
        """
        ИНН Исполнителя, КПП Исполнителя, Номер акта / упд / сф, договор
        """
        # Camelot extraction (from pdf(dataTable) to csv)
        camelot_instance = PdfCamelot(path_dir='extract_assets/input_files/upds_and_invoices',
                                      pdf_file='Передаточный документ 31.05.24 № 54503 = 2 191.99 без НДС.pdf')
        # Считываем таблицы с помощью камелота
        tables = camelot_instance.read_tables()
        # Нас интересует только первая таблица (данные по товарам, camelot выделяет первым эту часть данных)
        table: list = tables[0].data
        # Затем импортируем класс DataCleaning для очистки и работы с извлеченной таблицей
        cleaned_table: list = DataCleaning.data_clean(data_table=table)
        print(cleaned_table)

        camelot_instance.write_to_csv(tables=tables, file_csv_name='Camelot_result.csv')

        # Re extraction (inn/kpp and invoice) to json
        pdf = PdfTextReader(path_dir='extract_assets/input_files/upds_and_invoices',
                            pdf_file='УПД 31.05.24 № 428 = 257 428.00 без НДС.pdf')
        # Извлекаем текст из pdf
        text = pdf.extract_text_from_pdf()

        my_regulars: 'InnInvoiceDataExtraction' = InnInvoiceDataExtraction(text=text)
        logger.info('Извлеченный текст документа:\n\n%s\n', my_regulars.text)

        inn_kpp: str = my_regulars.inn_and_kpp_extract()
        invoice: str = my_regulars.invoice_extract()
        contract_number: str = my_regulars.contract_extract()

        # Должно идти только после очистки таблицы данных
        data = my_regulars.data_collect(inn_kpp=inn_kpp, invoice=invoice, contract_number=contract_number,
                                        data_table=table)
        logger.info('Хэш-таблица версия 1 (до очистки): \n%s', data)

        if type(data) == dict:
            DictToJson.write_to_json(collection=data)
        logger.info('---------------Execution time: %s---------------', f'{(time.time() - start):.2f} seconds')

        # print(data, '---------------Execution time---------------', f'{(time.time() - start):.2f} seconds', sep='\n')
