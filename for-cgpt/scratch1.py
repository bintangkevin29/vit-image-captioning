# %%
#python 3.10.12
from IPython.display import clear_output
# %pip install datasets evaluate rouge_score torch sacremoses -q
# %pip install transformers==4.28.0
# %pip install -U accelerator
# %pip install -U transformers[torch] transformers[sentencepiece]
clear_output()

# %%
model_name = 'vit-gpt-en_v0.0'

model_dir = f'./models/{model_name}'
model_output_dir = f'{model_dir}/image-captioning-output'
output_dir = f'{model_dir}/vit-gpt-model'

# %%
import os
os.environ["WANDB_DISABLED"] = "true"

# %%
import nltk
try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    nltk.download("punkt", quiet=True)

# %%
from transformers import VisionEncoderDecoderModel, AutoTokenizer, AutoFeatureExtractor
import torch

image_encoder_model = "google/vit-base-patch16-224-in21k"
text_decode_model = "gpt2"

model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
    image_encoder_model, text_decode_model)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
model.to(device)
clear_output()


# %%
# image feature extractor
feature_extractor = AutoFeatureExtractor.from_pretrained(image_encoder_model)
# text tokenizer
tokenizer = AutoTokenizer.from_pretrained(text_decode_model)

# %%
tokenizer.pad_token = tokenizer.eos_token

model.config.eos_token_id = tokenizer.eos_token_id
model.config.decoder_start_token_id = tokenizer.bos_token_id
model.config.pad_token_id = tokenizer.pad_token_id

# %%
model.save_pretrained(output_dir)
feature_extractor.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

# %%
from datasets import load_dataset, Dataset, DatasetDict
ds = load_dataset("jxie/flickr8k") 

raw_ds_train = Dataset.from_dict(ds['train'][0:5917])
raw_ds_test = Dataset.from_dict(ds['test'][0:250])
raw_ds_validation = Dataset.from_dict(ds['validation'][0:506])

ds = DatasetDict({
    "train": raw_ds_train,
    "test": raw_ds_test,
    "validation": raw_ds_validation
})

# %%
ds['train'][0]

# %%
from PIL import Image
import torch


class ImageCapatioingDataset(torch.utils.data.Dataset):
    def __init__(self, ds, ds_type, max_target_length):
        self.ds = ds
        self.max_target_length = max_target_length
        self.ds_type = ds_type

    def __getitem__(self, idx):
        image_path = self.ds[self.ds_type]['image_path'][idx]
        caption = self.ds[self.ds_type]['caption'][idx]
        model_inputs = dict()
        model_inputs['labels'] = self.tokenization_fn(caption, self.max_target_length)
        model_inputs['pixel_values'] = self.feature_extraction_fn(image_path)
        return model_inputs

    def __len__(self):
        return len(self.ds[self.ds_type])
    
    # text preprocessing step
    def tokenization_fn(self, caption, max_target_length):
        labels = tokenizer(caption, 
                          padding="max_length", 
                          max_length=max_target_length,
                          truncation=True).input_ids

        return labels
    
    # image preprocessing step
    def feature_extraction_fn(self, image_path):
        image = Image.open(image_path).convert('RGB')

        encoder_inputs = feature_extractor(images=image, return_tensors="np")

        return encoder_inputs.pixel_values[0]


train_ds = ImageCapatioingDataset(ds, 'train', 64)
eval_ds = ImageCapatioingDataset(ds, 'validation', 64)

# %%
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    predict_with_generate=True,
    evaluation_strategy="epoch",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    output_dir=model_output_dir,
)

# %%
import evaluate
metric = evaluate.load("rouge")

# %%
import numpy as np

ignore_pad_token_for_loss = True


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    if ignore_pad_token_for_loss:
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds,
                                                     decoded_labels)

    result = metric.compute(predictions=decoded_preds,
                            references=decoded_labels,
                            use_stemmer=True)
    result = {k: round(v * 100, 4) for k, v in result.items()}
    prediction_lens = [
        np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds
    ]
    result["gen_len"] = np.mean(prediction_lens)
    return result

# %%
from transformers import default_data_collator

# instantiate trainer
trainer = Seq2SeqTrainer(
    model=model,
    tokenizer=feature_extractor,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    data_collator=default_data_collator,
)

# %%
trainer.train()

# %%
trainer.save_model(model_output_dir)

# %%
tokenizer.save_pretrained(model_output_dir)

# %%
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image

model = VisionEncoderDecoderModel.from_pretrained(model_output_dir)
feature_extractor = ViTImageProcessor.from_pretrained(model_output_dir)
tokenizer = AutoTokenizer.from_pretrained(model_output_dir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

max_length = 16
num_beams = 4
gen_kwargs = {
    "max_length": max_length,
    "num_beams": num_beams,
    # "num_return_sequences": 3,
}


def predict_step(image_paths):
    images = []
    for image_path in image_paths:
        i_image = Image.open(image_path)
        if i_image.mode != "RGB":
            i_image = i_image.convert(mode="RGB")

        images.append(i_image)

    pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    output_ids = model.generate(pixel_values, **gen_kwargs)

    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]
    for i, pred in enumerate(preds): 
        showImg = Image.open(image_paths[i]).resize((300,300))
        display(showImg)
        print(pred.strip())


# %%
import glob

image_paths = []
for filename in glob.glob('./image-samples/*.jpg'): 
  image_paths.append(filename)

predict_step(image_paths)


