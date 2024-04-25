import re

text = '''
hps SystemTechnik GmbH - Altdorfer StraBe 16 - 88276 Berg

SPRINTIS Schenk GmbH & Co. KG_ONLINE
Bestell-Team

Ludwig-Weis-StraRe 11

97082 Wurzburg

Bestellung an Lieferant BL24-10003
Rechnung bitte 2-fach

Zahlungsbedingung: paypal

Sehr geehrtes Bestell-Team,

SystemTechnik

ite)

Competence in Training

Sachbearbeiter: Team Einkauf

Telefon: (0751) 56075-10

Telefax:

E-mail: einkauf@hps-systemtechnik.com

Unsere Kundennr.: 80117
Lieferant: 80117/80117

Ihre Tel.-Nr.: (0931) 40416222
Ihre Fax Nr.:

Datum: 11.4.2024

wir bestellen zu den Allgemeinen Lieferbedingungen flr Leistungen und Erzeugnisse der Elektroindustrie (Stand
2

Jan. 2002).

Pos. Artikel Nr.

1. 1 Rolle
GZ.RL.10.02.90.593

Artikel-Bezeichnung

Z002878

Gummiband schwarz 10 mm breit
Gummilitze 10 mm 12 fdg.
Polyester - Elasthodien
Artikelnummer: 49018/10

Farbe: 7010/schwarz

VE: 200m pro Spule

hps SystemTechnik
Lehr- + Lernmittel GmbH
Altdorfer Strake 16
88276 Berg

USt-IdNr.: DE119820017

Tel.: (0751) 56075-0

Fax: (0751) 56075-77
E-Mail:support@hps-systemtechnik.com
Web: www.hps-systemtechnik.com

Menge/ _Liefer- E-Preis € Rabatt G-Preis
Einheit Termin
90,00 KW 17 0,23 20,7
m

Warenwert € 20,70

Steuer 19% € 3,93

Gesamtsumme € 24,63

Seite - 1 -
Geschaftsfuhrer: National-Bank Milheim AG, Essen

Dr.-Ing.Sergej Bagh Konto-Nr. 168041, BLZ 36020030
Sitz der Gesellschaft: Berg
Handelsregister Ulm Nr. ARB 552598
'''

# pattern = r'(\d+,\d+)'
# result = re.search(pattern, text, re.DOTALL)
#
# if result:
#     print(result.group())
# else:
#     print("Match not found")


class PatternDataExtraction:
    def __init__(self, txt: str):
        self.txt = txt
        self.extracted_data = {}

    def extract_article_number(self) -> dict:
        article_pattern = r"(Z\d{6})"       # Регулярка для извлечение номера артикула
        result = re.search(article_pattern, self.txt)
        self.extracted_data['article_number'] = result.group(0)
        return self.extracted_data

    def extract_quantity(self) -> dict:
        quantity_pattern = r'(\d+,\d+)'     # Регулярка для извлечения кол-ва (menge)
        result = re.search(quantity_pattern, self.txt, re.DOTALL)
        self.extracted_data['quantity'] = result.group(0)
        return self.extracted_data
