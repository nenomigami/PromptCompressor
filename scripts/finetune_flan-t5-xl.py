from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer



def train(train_file_path,
          model_name,
          output_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast = False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    def concat_instruction_input(examples):
        if examples["input"]:
            source = f"Instruction: {examples['instruction']}\nInput: {examples['input']}\nOutput: \n"
        else:
            # No input, instruction only.
            source = f"Instruction: {examples['instruction']}\nOutput: \n"
        model_inputs = tokenizer(source, max_length=512, truncation=True)
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(examples["output"], max_length=128, truncation=True)
        model_inputs["labels"] = labels.input_ids
        return model_inputs   

    train_dataset = load_dataset(train_file_path, split="train")
    train_dataset = train_dataset.map(concat_instruction_input, batched=False,remove_columns=["instruction", "input", "split", "source", "output"], num_proc=8)

    tokenizer.save_pretrained(output_dir)
        
    model_cls = AutoModelForSeq2SeqLM if "t5" in model_name else AutoModelForCausalLM
    model = model_cls.from_pretrained(model_name) 

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    model.save_pretrained(output_dir)

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=1,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model()

if __name__=="__main__":
    train_file_path = "data/alpaca_plus.py"
    model_name = "google/flan-t5-xl"
    output_dir = 'flan-t5-xl_finetuned'
    train(
        train_file_path=train_file_path,
        model_name=model_name,
        output_dir=output_dir,
    )