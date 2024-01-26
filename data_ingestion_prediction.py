import os
from PyPDF2 import PdfReader, PdfWriter
import fitz
import pandas as pd
from data_ingestion import Data_ingestion

text_data = Data_ingestion("doc_3.pdf","doc_3_inf","doc3_txt","token_doc_3",512,"admin.json")
text_data.splitting_file_into_pages()
text_data.converting_pdf_into_text_file()
print("text file created")



