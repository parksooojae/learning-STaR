def star_iteration(iteration: int):
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, ...)
    
    # ... fine-tune ...
    
    # Save locally (overwrite)
    model.save_pretrained(MODEL_PATH)
    tokenizer.save_pretrained(MODEL_PATH)
    
    # Backup to Hub (free cloud storage)
    model.push_to_hub(f"your-username/star-iter-{iteration}")



