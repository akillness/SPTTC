
## Fine-tune a pretrained model

> Introduction : <https://huggingface.co/docs/transformers/main/en/training>

> Model Hub : <https://huggingface.co/models>

There are significant benefits to using a pretrained model. It reduces computation costs, your carbon footprint, and allows you to use state-of-the-art models without having to train one from scratch. 🤗 Transformers provides access to thousands of pretrained models for a wide range of tasks. When you use a pretrained model, you train it on a dataset specific to your task. This is known as fine-tuning, an incredibly powerful training technique. In this tutorial, you will fine-tune a pretrained model with a deep learning framework of your choice:

- Fine-tune a pretrained model with 🤗 Transformers [Trainer](https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.Trainer).
- Fine-tune a pretrained model in TensorFlow with Keras.
- Fine-tune a pretrained model in native PyTorch.

### Prepare a dataset

[![Prepare a dataset](http://img.youtube.com/vi/_BZearw7f0w/0.jpg)](https://youtu.be/_BZearw7f0w)

Before you can fine-tune a pretrained model, download a dataset and prepare it for training. The previous tutorial showed you how to process data for training, and now you get an opportunity to put those skills to the test!

As you now know, you need a tokenizer to process the text and include a padding and truncation strategy to handle any variable sequence lengths. To process your dataset in one step, use 🤗 Datasets map method to apply a preprocessing function over the entire dataset:

~~~py
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")


def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


tokenized_datasets = dataset.map(tokenize_function, batched=True)
~~~~

If you like, you can create a smaller subset of the full dataset to fine-tune on to reduce the time it takes:

~~~py
small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))
~~~

### Train

At this point, you should follow the section corresponding to the framework you want to use. You can use the links in the right sidebar to jump to the one you want - and if you want to hide all of the content for a given framework, just use the button at the top-right of that framework’s block!

[![Train](http://img.youtube.com/vi/nvBXf7s7vTI/0.jpg)](https://youtu.be/nvBXf7s7vTI)

#### Train with PyTorch Trainer

🤗 Transformers provides a Trainer class optimized for training 🤗 Transformers models, making it easier to start training without manually writing your own training loop. The Trainer API supports a wide range of training options and features such as logging, gradient accumulation, and mixed precision.

Start by loading your model and specify the number of expected labels. From the Yelp Review dataset card, you know there are five labels:

~~~py
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-cased", num_labels=5)

~~~

> [!Note]
> You will see a warning about some of the pretrained weights not being used and some weights being randomly initialized. Don’t worry, this is completely normal! The pretrained head of the BERT model is discarded, and replaced with a randomly initialized classification head. You will fine-tune this new model head on your sequence classification task, transferring the knowledge of the pretrained model to it.


#### Training hyperparameters

Next, create a TrainingArguments class which contains all the hyperparameters you can tune as well as flags for activating different training options. For this tutorial you can start with the default training hyperparameters, but feel free to experiment with these to find your optimal settings.

Specify where to save the checkpoints from your training:
~~~py
from transformers import TrainingArguments

training_args = TrainingArguments(output_dir="test_trainer")
~~~

#### Evaluate

Trainer does not automatically evaluate model performance during training. You’ll need to pass Trainer a function to compute and report metrics. The 🤗 Evaluate library provides a simple accuracy function you can load with the evaluate.load (see this quicktour for more information) function:

~~~py
import numpy as np
import evaluate

metric = evaluate.load("accuracy")
~~~

Call compute on metric to calculate the accuracy of your predictions. Before passing your predictions to compute, you need to convert the logits to predictions (remember all 🤗 Transformers models return logits):
~~~py
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)
~~~

If you’d like to monitor your evaluation metrics during fine-tuning, specify the eval_strategy parameter in your training arguments to report the evaluation metric at the end of each epoch:

~~~py
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(output_dir="test_trainer", eval_strategy="epoch")
~~~

#### Trainer

Create a [Trainer](https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.Trainer) object with your model, training arguments, training and test datasets, and evaluation function:
~~~py
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)
~~~

Then fine-tune your model by calling [train()](https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.Trainer.train):

~~~py
trainer.train()
~~~


* * *

## Train in native PyTorch

[![Train in native pytorch](http://img.youtube.com/vi/Dh9CL8fyG80/0.jpg)](https://youtu.be/Dh9CL8fyG80)

[Trainer](https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.Trainer) takes care of the training loop and allows you to fine-tune a model in a single line of code. For users who prefer to write their own training loop, you can also fine-tune a 🤗 Transformers model in native PyTorch.

At this point, you may need to restart your notebook or execute the following code to free some memory:
~~~py
del model
del trainer
torch.cuda.empty_cache()
~~~

Next, manually postprocess tokenized_dataset to prepare it for training.

1. Remove the text column because the model does not accept raw text as an input:
~~~py
tokenized_datasets = tokenized_datasets.remove_columns(["text"])
~~~

2. Rename the label column to labels because the model expects the argument to be named labels:

~~~py
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
~~~

3. Set the format of the dataset to return PyTorch tensors instead of lists:
~~~py
tokenized_datasets.set_format("torch")
~~~


Then create a smaller subset of the dataset as previously shown to speed up the fine-tuning:
~~~py
small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))
~~~

### Dataloader

Create a DataLoader for your training and test datasets so you can iterate over batches of data:

~~~py
from torch.utils.data import DataLoader

train_dataloader = DataLoader(small_train_dataset, shuffle=True, batch_size=8)
eval_dataloader = DataLoader(small_eval_dataset, batch_size=8)
~~~

Load your model with the number of expected labels:
~~~py
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-cased", num_labels=5)
~~~

### Optimizer and learning rate scheduler

Create an optimizer and learning rate scheduler to fine-tune the model. Let’s use the AdamW optimizer from PyTorch:
~~~py
from torch.optim import AdamW

optimizer = AdamW(model.parameters(), lr=5e-5)
~~~

Create the default learning rate scheduler from [Trainer](https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.Trainer):

~~~py
from transformers import get_scheduler

num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)
~~~

Lastly, specify device to use a GPU if you have access to one. Otherwise, training on a CPU may take several hours instead of a couple of minutes.

~~~py
import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
~~~

Great, now you are ready to train! 🥳


### Training loop

To keep track of your training progress, use the tqdm library to add a progress bar over the number of training steps:

~~~py
from tqdm.auto import tqdm

progress_bar = tqdm(range(num_training_steps))

model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

~~~

### Evaluate
Just like how you added an evaluation function to Trainer, you need to do the same when you write your own training loop. But instead of calculating and reporting the metric at the end of each epoch, this time you’ll accumulate all the batches with add_batch and calculate the metric at the very end.

~~~py
import evaluate

metric = evaluate.load("accuracy")
model.eval()
for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])

metric.compute()
~~~


### Additional resources
For more fine-tuning examples, refer to:

🤗 [Transformers Examples](https://github.com/huggingface/transformers/tree/main/examples) includes scripts to train common NLP tasks in PyTorch and TensorFlow.

🤗 [Transformers Notebooks](https://huggingface.co/docs/transformers/main/en/notebooks) contains various notebooks on how to fine-tune a model for specific tasks in PyTorch and TensorFlow.

* * *

### Preprocess
Before you can train a model on a dataset, it needs to be preprocessed into the expected model input format. Whether your data is text, images, or audio, they need to be converted and assembled into batches of tensors. 🤗 Transformers provides a set of preprocessing classes to help prepare your data for the model. In this tutorial, you’ll learn that for:

- Text, use a [Tokenizer](https://huggingface.co/docs/transformers/main/en/main_classes/tokenizer) to convert text into a sequence of tokens, create a numerical representation of the tokens, and assemble them into tensors.
- Speech and audio, use a [Feature extractor](https://huggingface.co/docs/transformers/main/en/main_classes/feature_extractor) to extract sequential features from audio waveforms and convert them into tensors.
- Image inputs use a [ImageProcessor](https://huggingface.co/docs/transformers/main/en/main_classes/image_processor) to convert images into tensors.
- Multimodal inputs, use a [Processor](https://huggingface.co/docs/transformers/main/en/main_classes/processors) to combine a tokenizer and a feature extractor or image processor.