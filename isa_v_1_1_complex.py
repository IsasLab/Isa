from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import speech_recognition as sr
from gtts import gTTS
import os
import pygame
import torch

class IsaAI:
    def __init__(self, use_pipeline=True):
        self.recognizer = sr.Recognizer()
        pygame.mixer.init()
        
        print("Initializing BLOOM model. This may take several minutes and requires significant RAM...")
        
        if use_pipeline:
            self.pipe = pipeline("text-generation", model="bigscience/bloom", device_map="auto")
            print("Using BLOOM pipeline for text generation.")
        else:
            self.tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom")
            self.model = AutoModelForCausalLM.from_pretrained("bigscience/bloom", device_map="auto")
            print("Using direct model loading for BLOOM.")
        
        self.use_pipeline = use_pipeline
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        print("Isa AI with full BLOOM model is ready!")

    def listen(self):
        with sr.Microphone() as source:
            print("Listening...")
            audio = self.recognizer.listen(source)
        try:
            text = self.recognizer.recognize_google(audio)
            print(f"You said: {text}")
            return text
        except sr.UnknownValueError:
            print("Sorry, I didn't catch that.")
            return None
        except sr.RequestError:
            print("Sorry, there was an error with the speech recognition service.")
            return None

    def speak(self, text):
        print(f"Isa: {text}")
        tts = gTTS(text=text, lang='en')
        filename = "response.mp3"
        tts.save(filename)
        
        pygame.mixer.music.load(filename)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
        
        pygame.mixer.music.unload()
        os.remove(filename)

    def generate_response(self, prompt):
        if self.use_pipeline:
            response = self.pipe(prompt, max_new_tokens=50, num_return_sequences=1, temperature=0.7, top_k=50, top_p=0.95, do_sample=True, no_repeat_ngram_size=2)[0]['generated_text']
        else:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=50,
                    num_return_sequences=1,
                    temperature=0.7,
                    top_k=50,
                    top_p=0.95,
                    do_sample=True,
                    no_repeat_ngram_size=2
                )
            response = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        return response.strip()

    def chat(self):
        self.speak("Hello, I am Isa, powered by the full BLOOM model. How can I assist you today?")
        
        conversation_history = []
        while True:
            user_input = self.listen()
            if user_input is None:
                continue
            
            if user_input.lower() in ['exit', 'quit', 'bye']:
                self.speak("Thank you for the conversation. Goodbye!")
                break
            
            conversation_history.append(f"Human: {user_input}")
            full_prompt = "\n".join(conversation_history) + "\nIsa:"
            response = self.generate_response(full_prompt)
            
            # Extract Isa's response from the generated text
            isa_response = response.split("Isa:")[-1].strip()
            
            conversation_history.append(f"Isa: {isa_response}")
            if len(conversation_history) > 10:  # Keep last 5 exchanges
                conversation_history = conversation_history[-10:]
            
            self.speak(isa_response)

if __name__ == "__main__":
    use_pipeline = True  # Set to False if you want to use direct model loading
    isa = IsaAI(use_pipeline)
    isa.chat()