import re
import json
from logging import getLogger

logger = getLogger(__name__)        # __name__ - имя модуля


class PatternDataExtraction:
    def __init__(self, txt: str):
        self.txt: str = txt

        self.article_numbers: list = []
        self.quantity_numbers: list = []
        self.term_delivery_numbers: list = []

        self.data_collection: dict = {}

    def extract_article_number(self) -> list:
        article_pattern = r"(Z\d{6})"       # Регулярка для извлечения номера артикула
        result = re.finditer(article_pattern, self.txt)

        for article in result:
            self.article_numbers.append(article.group(0))

        # self.extracted_data['article_number'] = articles
        return self.article_numbers

    def extract_quantity(self) -> list:
        quantity_pattern = r'\d{1,9},\d{2,3} KW'    # Регулярка для извлечения кол-ва (menge)
        result = re.finditer(quantity_pattern, self.txt, re.DOTALL)

        for quantity in result:
            self.quantity_numbers.append(quantity.group(0)[:2])

        # self.extracted_data['quantity'] = quantity_values
        return self.quantity_numbers

    def extract_term_delivery(self) -> list:
        term_delivery_pattern = r'KW\s+(\d+)'      # Регулярка для извлечения Liefer-Termin
        result = re.finditer(term_delivery_pattern, self.txt, re.DOTALL)

        for term in result:
            self.term_delivery_numbers.append(term.group(0)[3:])

        # self.extracted_data['term_delivery'] = term_delivery_values
        return self.term_delivery_numbers

    def data_collect(self, articles: list, quantities: list, terms_delivery: list) -> dict:
        self.data_collection['article_number'] = articles
        self.data_collection['quantity'] = quantities
        self.data_collection['term_delivery'] = terms_delivery

        return self.data_collection


class InnInvoiceDataExtraction:
    def __init__(self, text: str):
        self._text = text

        self.inn_kpp_seller: str = ''
        self.invoices: list = []
        self.data_collection: dict = {}

    @property
    def text(self) -> str:
        return self._text

    def inn_and_kpp_extract(self) -> str:
        pattern_inn_kpp = r'\d{10}/\d{9}'
        result = re.search(pattern_inn_kpp, self._text.replace(' ', ''), re.DOTALL)
        self.inn_kpp_seller = result.group(0)
        logger.info("ИНН/КПП %s", self.inn_kpp_seller)
        return self.inn_kpp_seller
    
    def invoice_extract(self) -> list:
        # Регулярное выражение для извлечения номера Счет-фактуры
        # pattern_invoice_number = re.compile(r'Счет-фактура№(\d{7}/\d{4}) от (\d{2}.\d{2}.\d{4})')

        pattern_invoice_number = re.compile(r'\d+/\d+|\d+')
        invoice_number_matches = pattern_invoice_number.findall(self._text)
        for matches in invoice_number_matches:
            self.invoices.append((matches[0]))
        logger.info("Счет-фактура документа: %s", self.invoices)
        return self.invoices
    
    def data_collect(self, inn_kpp_seller: str, invoices: list) -> dict | Exception:
        # print(f'Инн кпп продавца: {inn_kpp_seller}\nНомер счет фактуры: {invoices}')
        self.data_collection['inn_kpp_seller'] = inn_kpp_seller
        try:
            self.data_collection['invoice'] = invoices[0]
            logger.info('JSON data:\n%s', self.data_collection)
            return self.data_collection
        except IndexError as e:
            logger.error('Ошибка - %s', e)
            return e


class DictToJson:
    @staticmethod
    def write_to_json(collection: dict) -> None:
        with open('extracted_results/data.json', 'w') as file:
            json.dump(collection, file)
