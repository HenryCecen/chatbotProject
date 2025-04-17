import random
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from Helper.model import NeuralNet
from Helper.nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load intents
with open('Data/intents.json', 'r') as json_data:
    intents = json.load(json_data)

# Load custom-trained intent classification model
FILE = "Data/data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

intent_model = NeuralNet(input_size, hidden_size, output_size).to(device)
intent_model.load_state_dict(model_state)
intent_model.eval()

# Load DialoGPT model
dialog_model_name = "microsoft/DialoGPT-medium"
dialog_tokenizer = AutoTokenizer.from_pretrained(dialog_model_name)
dialog_model = AutoModelForCausalLM.from_pretrained(dialog_model_name)
dialog_model.eval()

bot_name = "Sam"
print("Bot is ready! Type 'quit' to exit.")

# DialoGPT response function
def generate_fallback_response(prompt):
    inputs = dialog_tokenizer.encode(prompt + dialog_tokenizer.eos_token, return_tensors='pt')
    with torch.no_grad():
        outputs = dialog_model.generate(
            inputs,
            max_length=100,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            pad_token_id=dialog_tokenizer.eos_token_id,
            temperature=0.8,
            top_k=50,
            top_p=0.9,
            do_sample=True
        )
    response = dialog_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response[len(prompt):].strip()

# Chat loop
while True:
    sentence = input("You: ")
    if sentence.lower() == "quit":
        print("Goodbye!")
        break

    # intent prediction
    tokens = tokenize(sentence)
    X = bag_of_words(tokens, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    with torch.no_grad():
        output = intent_model(X)
        _, predicted = torch.max(output, dim=1)
        tag = tags[predicted.item()]
        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]

    # if confidence is high, intents will be used. Or else DialoGPT will be used
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                print(f"{bot_name}: {random.choice(intent['responses'])}")
    else:
        # fallback to DialoGPT
        chatbot_response = generate_fallback_response(sentence)
        print(f"{bot_name}: {chatbot_response}")
