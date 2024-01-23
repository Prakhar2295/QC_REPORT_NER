import os
from PyPDF2 import PdfReader, PdfWriter
import fitz
import pandas as pd

class Data_ingestion:
	def __init__(self,file:str,destination_path:str,text_file_path:str,tokens_file_path:str,window_length:int,annotated_file_path:str):
		self.file = file
		self.destination_path = destination_path
		self.text_file_path = text_file_path
		self.tokens_file_path = tokens_file_path
		self.window_length = window_length
		self.annotated_file_path = annotated_file_path

	def splitting_file_into_pages(self):
		if self.file.endswith('.pdf'):
			self.pdf = PdfReader(self.file)
			if not os.path.isdir(self.destination_path):
				os.mkdir(self.destination_path)

			for page in range(len(self.pdf.pages)):  # Change 'reader' to 'pdf' here
				pdf_writer = PdfWriter()
				pdf_writer.add_page(self.pdf.pages[page])

				output_filename = '{}_page_{}.pdf'.format(self.file[:-4], page + 1)

				output_filepath = os.path.join(self.destination_path,output_filename)

				with open(output_filepath, 'wb') as out:
					pdf_writer.write(out)

				print('Created: {}'.format(output_filename))


	def converting_pdf_into_text_file(self):
		if self.text_file_path is not None:
			for i, file in enumerate(os.listdir(self.destination_path)):
				file_path = os.path.join(self.destination_path, file)
				pdf_document = fitz.open(file_path)

				# Assuming you want to read the text from the first page
				first_page = pdf_document[0]
				text = first_page.get_text("text")

				text_file_name = f"doc_{i}.txt"
				text_file_path_full = os.path.join(self.text_file_path, text_file_name)

				with open(text_file_path_full, 'w', encoding='utf-8') as text_file:
					text_file.write(text)


	def chunking_text_files_into_tokens(self):
		if self.text_file_path is not None:
			for i, file in enumerate(os.listdir(self.text_file_path)):
				file_path = os.path.join(self.text_file_path, file)
				with open(file_path, 'r') as f:
					text = f.read()

				total_len = len(text)
				#print(f"total_len:{total_len}")
				#window_length = 512
				start = 0

				while start < total_len:
					end = min(start + self.window_length, total_len)

					# Extract the text chunk correctly
					text_chunk = text[start:end]

					print(f"i_start:{i}_{start}")
					print(f"i_end:{i}_{end}")

					token_text_file_name = f"token_text_file_{i}_{start}_{end}.txt"
					token_file_path_full = os.path.join(self.tokens_file_path, token_text_file_name)

					with open(token_file_path_full, 'w', encoding='utf-8') as text_file:
						text_file.write(text_chunk)  # Write the correct variable

					start = end


	def dataframe_loader(self):
		if self.annotated_file_path is not None:
			self.df = pd.read_json(self.annotated_file_path,lines = True)
		return self.df




















