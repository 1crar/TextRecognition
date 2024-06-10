import re


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
        self._inn_kpp_numbers: list = []
        self._invoices: list = []

    def inn_and_kpp_extract(self) -> list:
        pattern_inn_kpp = re.compile(r'ИНН/КПП (\d{10}) / (\d{9})')
        inn_kpp_matches = pattern_inn_kpp.findall(self._text)

        # Вывод результатов
        for matches in inn_kpp_matches:
            self._inn_kpp_numbers.append(matches[0])
            self._inn_kpp_numbers.append(matches[1])

        return self._inn_kpp_numbers


