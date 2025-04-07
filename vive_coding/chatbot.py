import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

class SolarChatbot:
    def __init__(self):
        self.model_id = "upstage/TinySolar-248m-4k-py-instruct"
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, cache_dir='./cache')
        print("Loading model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            device_map="cpu",
            torch_dtype=torch.float32,
            cache_dir="./cache"
        )
        
    def generate_response(self, user_input, max_length=512):
        # Format the prompt according to the model's instruction format
        prompt = (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{user_input}\n\n### Response:\n"
        )
        
        # Tokenize the input
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                num_return_sequences=1,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode and clean up the response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the input prompt from the response
        response = response.replace(prompt, "").strip()
        
        return response

def main():
    print("Initializing Solar Chatbot...")
    chatbot = SolarChatbot()
    print("Chatbot initialized! Type 'quit' to exit.")
    print("-" * 50)
    
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() == 'quit':
            print("Goodbye!")
            break
            
        try:
            response = chatbot.generate_response(user_input)
            print("\nAssistant:", response)
        except Exception as e:
            print(f"\nError occurred: {str(e)}")

if __name__ == "__main__":
    main() 