import pandas as pd
import re
from data_ingestion import Data_ingestion


#df = Data_ingestion.dataframe_loader()

class data_preprocessing:
	def __init__(self):
		self.df = Data_ingestion.dataframe_loader()

	def convert_data_to_spacy_format(self):
		training_data = []
		for i in range(len(self.df)):
			text = self.df["text"][i].replace("\n", " ")
			# text.replace("\n"," ")
			label = self.df["label"][i]
			training_data.append([text, {"entities": label}])
		return training_data


	def remove_white_spaces_from_entities(self):

		data = self.convert_data_to_spacy_format()

		invalid_tokens = re.compile(r'\s')

		self.cleaned_data = []
		for text,annotations in data:
			cleaned_entities = []
			entities = annotations["entities"]

			for start,end,label in entities:
				valid_start = start
				valid_end = end

				while valid_start < len(text) and invalid_tokens.match(text[valid_start]):
					valid_start += 1

				while valid_end > 1 and invalid_tokens.match(text[valid_end -1]):
					valid_end -= 1

				cleaned_entities.append([valid_start,valid_end,label])
			self.cleaned_data.append([text,{"entities":cleaned_entities}])

		return self.cleaned_data


#data = remove_white_spaces_from_entities(a)
#print(f"cleaned_data:{data}")
#print(len(data[0][0]))

#a = convert_data_to_spacy_format(df)
#print(f"data:{a[0]}")
#print(len(a[0][0]))
