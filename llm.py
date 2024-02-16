import torch
from transformers import AutoTokenizer
from auto_gptq import AutoGPTQForCausalLM
from llama_cpp import Llama

class LLMAgent:
    def __init__(self):
        # model_name = "rinna/youri-7b-gptq"
        # model_name = "rinna/youri-7b-chat-gptq"
        model_name = "rinna/youri-7b-instruction-gptq"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoGPTQForCausalLM.from_quantized(model_name, use_safetensors=True)

    def create_instruction_prompt(self, text):
        instruction = "入力に対し適切な応答を行ってください。"
        instruction = "あなたの名前はとろろです。人間にフレンドリーなAIとして、入力に対し適切な応答を行ってください。"

        prompt = f"""
        以下は、タスクを説明する指示と、文脈のある入力の組み合わせです。要求を適切に満たす応答を書きなさい。

        ### 指示:
        {instruction}

        ### 入力:
        {text}

        ### 応答:
        """
        return prompt

    def gen(self, text):
        prompt = self.create_instruction_prompt(text)
        token_ids = self.tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=token_ids.to(self.model.device),
                max_new_tokens=20,
                # min_new_tokens=200,
                do_sample=True,
                temperature=1.0,
                top_p=0.95,
                pad_token_id=self.tokenizer.pad_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        output = self.tokenizer.decode(output_ids.tolist()[0])
        return output

def main():
    agent = ChatAgent()
    output = agent.gen("科学がテーマの雑談を作成してください。")
    # output = agent.gen(text)
    print(output)

if __name__ == '__main__':
    main()