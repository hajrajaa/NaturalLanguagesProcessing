#%%
import torch
import numpy as np
from datasets import load_dataset
from datasets import load_from_disk
from transformers import BertTokenizer 
from transformers import BertForSequenceClassification 
from transformers import TrainingArguments
from transformers import Trainer
from sklearn.metrics import accuracy_score




#define the tokenization function 
def tokenize_function(example):
    return tokenizer(example["text"], padding="max_length", truncation=True)




# define the Metric function 
def compute_metrics(eval_pred):
    logits,labels=eval_pred
    predictions=np.argmax(logits,axis=-1)
    return {"accuracy":accuracy_score(labels,predictions)}

# define the training function 
def train_function(train_dataset,eval_dataset,model):
    try:

        # define the training arguments
        train_args=TrainingArguments(
            output_dir="./results",
            logging_dir="./logs",
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=5,
            weight_decay=0.01,
            logging_steps=10,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            
        )

        #define the trainer 
        trainer=Trainer(
            model=model,
            args=train_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
        )

        # train the model
        trainer.train()

        return trainer,model



    except Exception as e:
        raise e






#%% 

if __name__=='__main__':
    try:


        # load the dataset 

        try:
            subset=load_from_disk("imdb_subset")
        except FileNotFoundError:

            dataset=load_dataset('imdb')


            subset=dataset["train"].shuffle(seed=42).select(range(500))

            subset.save_to_disk("imdb_subset")



       

        #subset=load_from_disk("imdb_subset")

        
             

        # load the Bert model 
        ## see how many labels should be used 
        model=BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)



        # load the tokenizer 
        tokenizer=BertTokenizer.from_pretrained("bert-base-uncased")

        tokenized_datasets=subset.map(tokenize_function, batched=True)

        tokenized_datasets=tokenized_datasets.remove_columns(["text"])


        #Change the column label to labels
        tokenized_datasets=tokenized_datasets.rename_column("label","labels")

        tokenized_datasets.set_format("torch")

        train_test_split=tokenized_datasets.train_test_split(test_size=0.2 , seed=42)

        train_dataset=train_test_split["train"]
        eval_dataset=train_test_split["test"]

        




        # Trining the model 
        trainer,model=train_function(train_dataset,eval_dataset,model)


        # evaluating the model 


        results=trainer.evaluate()
        print(results)
        # print(f"Test accuracy: {results['eval_accuracy']:.2f}")


        # Save the model
        #trainer.save_model("imdb_finetuned_model")

    

    
    except Exception as e:
        raise e







#%%

