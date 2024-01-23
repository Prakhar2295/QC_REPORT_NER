import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim import AdamW
from tqdm import trange
from transformers import BertForTokenClassification, BertTokenizerFast
from seqeval.metrics import classification_report
from dataloader import data_loader,get_special_tokens,annot_confusion_matrix,flat_accuracy


device = torch.device("cude" if torch.cuda.is_available() else "cpu")

train_dl,val_dl = data_loader()
TOKENIZER = BertTokenizerFast("D:/FSDS/NLP/Report_NER/Bert/vocab.txt",do_lower_case = True)

label_list = ["CERTIFICATE_NO","ISSUED_DATE","ORDER_NO","PURCHASE_ORDER","SUPPLIER_NAME","CUSTOMER_NAME","PLATE_NO","O","MISC"]
label2id = {v:k for k,v in enumerate(label_list)}

model_path = "D:/FSDS/NLP/Report_NER/Bert"
model = BertForTokenClassification.from_pretrained(model_path, num_labels=len(label2id))

model.to(device)
#pad_tok, sep_tok, cls_tok, o_lab = get_special_tokens(TOKENIZER,label2id)
optimizer = AdamW(model.parameters(),lr = 2e-5)

MAX_GRAD_NORM = 1.0

EPOCHS = 4

def train_and_save_model(
		model,
		tokenizer,
		optimizer,
		epochs,
		idx2tag,
		tag2idx,
		max_grad_norm,
		device,
		train_dataloader,
		valid_dataloader
):
	pad_tok, sep_tok, cls_tok, o_lab = get_special_tokens(tokenizer, label2id)

	epoch = 0
	val_acc = []
	ep = []
	val_loss = []
	for _ in trange(epochs, desc="Epoch"):
		epoch += 1

		# Training loop
		print("Starting training loop.")
		model.train()
		tr_loss, tr_accuracy = 0, 0
		nb_tr_examples, nb_tr_steps = 0, 0
		tr_preds, tr_labels = [], []

		for step, batch in enumerate(train_dataloader):
			# Add batch to gpu

			# batch = tuple(t.to(device) for t in batch)
			b_input_ids, b_input_mask, b_labels = batch['input_ids'], batch['attention_mask'], batch['labels']
			b_input_ids, b_input_mask, b_labels = b_input_ids.to(device), b_input_mask.to(device), b_labels.to(device)

			# Forward pass
			outputs = model(
				b_input_ids,
				token_type_ids=None,
				attention_mask=b_input_mask,
				labels=b_labels,
			)
			loss, tr_logits = outputs[:2]
			# print(f"outputs:{outputs[1].shape}")
			print(f"tr_logits:{tr_logits.size()}")

			# Backward pass
			loss.backward()

			# loss.sum.backward()

			# Compute train loss
			tr_loss += loss.item()
			nb_tr_examples += b_input_ids.size(0)
			nb_tr_steps += 1

			# Subset out unwanted predictions on CLS/PAD/SEP tokens
			preds_mask = (
					(b_input_ids != cls_tok)
					& (b_input_ids != pad_tok)
					& (b_input_ids != sep_tok)
			)

			tr_logits = tr_logits.cpu().detach().numpy()
			# print(f"b_labels:{b_labels.shape}")
			tr_label_ids = torch.masked_select(b_labels, (preds_mask == 1))
			preds_mask = preds_mask.cpu().detach().numpy()
			print(f"tr_preds_mask: {preds_mask.shape}")
			tr_batch_preds = np.argmax(tr_logits[preds_mask.squeeze()], axis=1)
			# print(f"pred_squeeze:{preds_mask.squeeze()}")
			# print(f"pred_squeeze_shape:{preds_mask.squeeze().shape}")
			# print(f"tr_pred_squeeze:{tr_logits[preds_mask.squeeze()]}")
			# print(f"tr_pred_squeeze_shape:{tr_logits[preds_mask.squeeze()].shape}")
			# print(f"tr_batch_preds:{tr_batch_preds}")
			tr_batch_labels = tr_label_ids.to("cpu").numpy()
			tr_preds.extend(tr_batch_preds)
			tr_labels.extend(tr_batch_labels)

			# Compute training accuracy
			tmp_tr_accuracy = flat_accuracy(tr_batch_labels, tr_batch_preds)
			tr_accuracy += tmp_tr_accuracy

			# Gradient clipping
			torch.nn.utils.clip_grad_norm_(
				parameters=model.parameters(), max_norm=max_grad_norm
			)

			# Update parameters
			optimizer.step()
			model.zero_grad()

		tr_loss = tr_loss / nb_tr_steps
		tr_accuracy = tr_accuracy / nb_tr_steps

		# Print training loss and accuracy per epoch
		print(f"Train loss: {tr_loss}")
		print(f"Train accuracy: {tr_accuracy}")

		"""
		Validation loop
		"""
		print("Starting validation loop.")

		model.eval()
		eval_loss, eval_accuracy = 0, 0
		nb_eval_steps, nb_eval_examples = 0, 0
		predictions, true_labels = [], []

		for batch in valid_dataloader:
			b_input_ids, b_input_mask, b_labels = batch['input_ids'], batch['attention_mask'], batch['labels']
			b_input_ids, b_input_mask, b_labels = b_input_ids.to(device), b_input_mask.to(device), b_labels.to(device)

			with torch.no_grad():
				outputs = model(
					b_input_ids,
					token_type_ids=None,
					attention_mask=b_input_mask,
					labels=b_labels,
				)
				tmp_eval_loss, logits = outputs[:2]

			# Subset out unwanted predictions on CLS/PAD/SEP tokens
			preds_mask = (
					(b_input_ids != cls_tok)
					& (b_input_ids != pad_tok)
					& (b_input_ids != sep_tok)
			)

			logits = logits.cpu().detach().numpy()
			print(f"pr_logits_shape: {logits.shape}")
			label_ids = torch.masked_select(b_labels, (preds_mask == 1))
			preds_mask = preds_mask.cpu().detach().numpy()
			print(f"pr_preds_mask_shape: {preds_mask.shape}")
			val_batch_preds = np.argmax(logits[preds_mask.squeeze()], axis=1)
			val_batch_labels = label_ids.to("cpu").numpy()
			predictions.extend(val_batch_preds)
			true_labels.extend(val_batch_labels)

			tmp_eval_accuracy = flat_accuracy(val_batch_labels, val_batch_preds)

			eval_loss += tmp_eval_loss.mean().item()
			eval_accuracy += tmp_eval_accuracy

			nb_eval_examples += b_input_ids.size(0)
			nb_eval_steps += 1

		# Evaluate loss, acc, conf. matrix, and class. report on devset
		pred_tags = [idx2tag[i] for i in predictions]
		valid_tags = [idx2tag[i] for i in true_labels]
		cl_report = classification_report([valid_tags], [pred_tags])
		conf_mat = annot_confusion_matrix(valid_tags, pred_tags)
		eval_loss = eval_loss / nb_eval_steps
		eval_accuracy = eval_accuracy / nb_eval_steps

		# Report metrics
		print(f"Validation loss: {eval_loss}")
		print(f"Validation Accuracy: {eval_accuracy}")
		print(f"Classification Report:\n {cl_report}")
		print(f"Confusion Matrix:\n {conf_mat}")
		val_acc.append(eval_accuracy)
		ep.append(epoch)
		val_loss.append(eval_loss)

	plt.plot(ep, val_acc, 'g', label='Validation accuracy')
	# plt.plot(ep, val_loss, 'b', label='Validation loss')
	plt.title('Validation acuuracy ')
	plt.xlabel('Epochs')
	plt.ylabel('Accuracy')
	plt.legend()
	plt.show()


#train_and_save_model(
    #model,
    #TOKENIZER,
    #optimizer,
    #7,
    #id2label,
    #label2id,
    #MAX_GRAD_NORM,
    #DEVICE,
    #train_dl,
    #val_dl
#)


torch.save(
    {
        "epoch": EPOCHS,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    },
    'model_e10.tar',
)