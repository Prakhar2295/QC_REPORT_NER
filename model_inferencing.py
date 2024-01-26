import torch
import numpy as np
from transformers import BertForTokenClassification, BertTokenizerFast
from dataloader import data_loader,get_special_tokens,annot_confusion_matrix,flat_accuracy



MAX_LEN = 512
EPOCHS = 4
DEVICE = torch.device("cpu")
MODEL_PATH = "D:/FSDS/NLP/Report_NER/Bert"
STATE_DICT = torch.load('/kaggle/input/trained-ner-model/model_e10.tar', map_location=DEVICE)
TOKENIZER = BertTokenizerFast('../input/bert-base-uncased/vocab.txt', lowercase=True)
MODEL = BertForTokenClassification.from_pretrained(MODEL_PATH, state_dict=STATE_DICT['model_state_dict'], num_labels=9)


label_list = ["CERTIFICATE_NO","ISSUED_DATE","ORDER_NO","PURCHASE_ORDER","SUPPLIER_NAME","CUSTOMER_NAME","PLATE_NO","O","MISC"]
id2label = {k:v for k,v in enumerate(label_list)}
label2id = {v:k for k,v in enumerate(label_list)}


model = MODEL
model.to(DEVICE)


class model_prediction:
	def __init__(self,data,max_len):
		self.data = data
		self.tokenzier = TOKENIZER
		self.max_len = max_len
		self.model = MODEL
		self.device =torch.device("cpu")

	def process_data(self):
		tok = self.tokenzier.encode_plus(self.data, max_length=self.max_len, return_offsets_mapping=True)

		curr_sent = dict()

		padding_length = self.max_len - len(tok['input_ids'])

		curr_sent['input_ids'] = tok['input_ids'] + ([0] * padding_length)
		curr_sent['token_type_ids'] = tok['token_type_ids'] + ([0] * padding_length)
		curr_sent['attention_mask'] = tok['attention_mask'] + ([0] * padding_length)

		final_data = {
			'input_ids': torch.tensor(curr_sent['input_ids'], dtype=torch.long),
			'token_type_ids': torch.tensor(curr_sent['token_type_ids'], dtype=torch.long),
			'attention_mask': torch.tensor(curr_sent['attention_mask'], dtype=torch.long),
			'offset_mapping': tok['offset_mapping'],
			'tokens': tok.tokens()
		}
		return final_data

	def predict(self, idx2tag, tag2idx):
		model.eval()
		data = self.process_data()
		# print(type(data["offset_mapping"]))
		# print(data["offset_mapping"])
		input_ids, input_mask = data['input_ids'].unsqueeze(0), data['attention_mask'].unsqueeze(0)
		labels = torch.tensor([1] * input_ids.size(0), dtype=torch.long).unsqueeze(0)
		with torch.no_grad():
			outputs = self.model(
				input_ids,
				token_type_ids=None,
				attention_mask=input_mask,
				labels=labels,
			)
			tmp_eval_loss, logits = outputs[:2]

		logits = logits.cpu().detach().numpy()
		# print(logits)
		label_ids = np.argmax(logits, axis=2)
		# print(type(label_ids))
		# print(label_ids)
		# print(len(label_ids[0]))

		final_data = []
		entities = []
		for label_id, offset in zip(label_ids[0], data['offset_mapping']):
			curr_id = idx2tag[label_id]
			curr_start = offset[0]
			curr_end = offset[1]
			if curr_id != 'O':
				if len(entities) > 0 and entities[-1]['entity'] == curr_id and curr_start - entities[-1]['end'] in [0,
				                                                                                                    1]:
					entities[-1]['end'] = curr_end
				else:
					entities.append({'entity': curr_id, 'start': curr_start, 'end': curr_end})
		#final_data.append([data["tokens"], entities, self.data])
		for ent in entities:
			ent['text'] = self.data[ent['start']:ent['end']]
		return entities










