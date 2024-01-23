import torch
from transformers import BertForTokenClassification,BertTokenizerFast
from torch.utils.data import Dataset,DataLoader,RandomSampler,SequentialSampler
from data_preprocessing import data_preprocessing



#data_path = "D:/FSDS/NLP/Report_NER/Annotated_report/admin.jsonl"

#df = pd.read_json(data_path,lines = True)


class data_preparation:
	def __init__(self):
		self.max_len = 512
		self.model_path = "D:/FSDS/NLP/Report_NER/Bert"
		self.tokenizer = BertTokenizerFast("D:/FSDS/NLP/Report_NER/Bert/vocab.txt",do_lower_case = True)
		self.device = torch.device("cude" if torch.cuda.is_available() else "cpu")
		#self.data = data_preprocessing.remove_white_spaces_from_entities()   ###data in the spacy format
		self.label_list = ["CERTIFICATE_NO","ISSUED_DATE","ORDER_NO","PURCHASE_ORDER","SUPPLIER_NAME","CUSTOMER_NAME","PLATE_NO","O","MISC"]
		self.label2id = {v:k for k,v in enumerate(self.label_list)}


	def get_label(self,offset,labels):
		if offset[0] == 0 and offset[1] == 0:
			return 'O'
		for label in labels:
			if offset[1] >= label[0] and offset[0] <= label[1]:
				return label[2]
		return 'O'


	def process_reports(self,data,is_test = False):
		tok = self.tokenizer.encode_plus(data[0],max_length=self.max_len,return_offsets_mapping= True)

		encoded_data = {'orig_labels': [],'labels': []}

		padding_length = self.max_len - len(tok['input_ids'])

		if not is_test:
			labels = self.data[1]["entities"]
			labels.reverse()
			for off in tok['offset_mapping']:
				label = self.get_label(off,labels)
				encoded_data['orig_labels'].append(label)
				encoded_data['labels'].append(self.label2id["label"])

			encoded_data["labels"] = encoded_data['labels'] + [0 * padding_length]

		encoded_data["input_ids"] = tok['input_ids'] = [0 * padding_length]
		encoded_data["token_type_ids"] = tok['token_type_ids'] = [0 * padding_length]
		encoded_data["attention_mask"] = tok['attention_mask'] = [0 * padding_length]

		return encoded_data

class ReportDataset(Dataset):
	def __init__(self,data,tokenizer,label2id,max_len,is_test = False):
		self.data = data
		self.tokenizer = tokenizer
		self.label2id = label2id
		self.max_len = max_len
		self.is_test = is_test
		self.data_preparation = data_preparation()
		#self.data = data_preprocessing.remove_white_spaces_from_entities()

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		data = self.data_preparation.process_reports(self.data)

		return {
			"input_ids": torch.tensor(data["input_ids"],dtype = torch.long),
			"token_type_ids":torch.tensor(data["token_type_ids"],dtype = torch.long),
			"attention_mask": torch.tensor(data["attention_mask"]),
			"labels":torch.tensor(data["labels"],dtype = torch.long),
			#"orig_label":data["orig_labels"]

		}










#total = len(clean_data)
#train_data,val_data = data[0:150],data[150:]


#train_dataset = ReportDataset(train_data,TOKENIZER,label2id,MAX_LEN)
#val_dataset = ReportDataset(val_data,TOKENIZER,label2id,MAX_LEN)


#train_sampler = RandomSampler(train_dataset)

#train_dataloader = DataLoader(train_dataset,sampler = train_sampler,batch_size = 4)
#val_dataloader = DataLoader(val_dataset,batch_size = 4)


#MAX_LEN = 512
#EPOCHS = 4
#MODEL_PATH = "D:/FSDS/NLP/Report_NER/Bert"
#TOKENIZER = BertTokenizerFast("D:/FSDS/NLP/Report_NER/Bert/vocab.txt",do_lower_case = True)
#DEVICE = torch.device("cude" if torch.cuda.is_available() else "cpu")


#label_list = ["CERTIFICATE_NO","ISSUED_DATE","ORDER_NO","PURCHASE_ORDER","SUPPLIER_NAME","CUSTOMER_NAME","PLATE_NO","O"]
#id2label = {k:v for k,v in enumerate(label_list)}
#label2id = {v:k for k,v in enumerate(label_list)}







#data = convert_data_to_spacy_format(df)
#clean_data = remove_white_spaces_from_entities(data)
#f = process_reports(clean_data,TOKENIZER,label2id,MAX_LEN,is_test = False)
#print(f)





