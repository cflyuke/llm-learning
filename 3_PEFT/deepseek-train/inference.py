from modelscope import AutoModelForCausalLM, AutoTokenizer
from utils import SYSTEM_PROMPT

def infer(args):
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint_dir,
        dtype="auto",
        device_map="auto"
    )

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_dir)

    while True:
        print("你: ", end='')
        prompt = input()
        if prompt in ("exit", "bye"):
            print("Assistant: 再见👋")
            break

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
        
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer(text, return_tensors="pt").to(model.device)

        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=args.max_completion_length
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        print(f"Assistant: {response}")