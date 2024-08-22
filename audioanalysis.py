import whisper
import torch
import nltk
from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from collections import Counter
from anthropic import Anthropic
from dotenv import load_dotenv
import os
import logging
from pathlib import Path

# Load environment variables from .env file
load_dotenv()

# Download necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('maxent_ne_chunker', quiet=True)
nltk.download('words', quiet=True)
nltk.download('stopwords', quiet=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def transcribe_audio(audio_file, model_name="base", language=None):
    try:
        logging.info(f"Loading Whisper model: {model_name}")
        model = whisper.load_model(model_name)
        
        logging.info(f"Transcribing {audio_file}...")
        logging.info(f"File exists: {os.path.exists(audio_file)}")
        logging.info(f"File size: {os.path.getsize(audio_file)} bytes")
        
        logging.info("Using CPU for transcription")
        
        # Force CPU usage
        with torch.no_grad():
            if language:
                result = model.transcribe(audio_file, language=language, fp16=False)
            else:
                result = model.transcribe(audio_file, fp16=False)
        
        logging.info("Transcription complete!")
        return result["text"]
    except Exception as e:
        logging.error(f"Error during transcription: {str(e)}", exc_info=True)
        return None

def verify_cuda():
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name()}")
    else:
        print("CUDA is not available. The script will run on CPU, which may be slower.")

def transcribe_audio(audio_file, model_name="base", language=None):
    try:
        print(f"Loading Whisper model: {model_name}")
        model = whisper.load_model(model_name)
        
        print(f"Transcribing {audio_file}...")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        if language:
            result = model.transcribe(audio_file, language=language, fp16=torch.cuda.is_available(), device=device)
        else:
            result = model.transcribe(audio_file, fp16=torch.cuda.is_available(), device=device)
        
        print("Transcription complete!")
        return result["text"]
    except Exception as e:
        print(f"Error during transcription: {str(e)}")
        return None
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("CUDA cache cleared.")

def process_text(text):
    # Tokenization
    tokens = word_tokenize(text)
    
    # Part-of-speech tagging
    pos_tags = pos_tag(tokens)
    
    # Named Entity Recognition
    ner_tree = ne_chunk(pos_tags)
    
    # Extract named entities
    named_entities = []
    for chunk in ner_tree:
        if hasattr(chunk, 'label'):
            named_entities.append(' '.join(c[0] for c in chunk))
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word.lower() for word in tokens if word.isalnum() and word.lower() not in stop_words]
    
    # Calculate word frequencies
    word_freq = FreqDist(filtered_tokens)
    
    # Get the most common words
    most_common_words = word_freq.most_common(10)
    
    # Calculate sentence complexity (average words per sentence)
    sentences = text.split('.')
    words_per_sentence = sum(len(sentence.split()) for sentence in sentences) / len(sentences)
    
    # Identify potential key phrases (bigrams)
    bigrams = list(nltk.bigrams(filtered_tokens))
    bigram_freq = Counter(bigrams)
    common_bigrams = bigram_freq.most_common(5)
    
    # Prepare the processed data
    processed_data = {
        'original_text': text,
        'tokens': tokens,
        'pos_tags': pos_tags,
        'named_entities': named_entities,
        'word_frequency': dict(word_freq),
        'most_common_words': most_common_words,
        'words_per_sentence': words_per_sentence,
        'common_bigrams': common_bigrams
    }
    
    return processed_data

def analyze_with_llm(processed_text, question):
    # Initialize the Anthropic client
    anthropic = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    # Prepare the context for Claude
    context = f"""
    Original Text: {processed_text['original_text']}

    Named Entities: {', '.join(processed_text['named_entities'])}

    Most Common Words: {', '.join(f"{word}({count})" for word, count in processed_text['most_common_words'])}

    Common Bigrams: {', '.join(f"{' '.join(bigram)}({count})" for bigram, count in processed_text['common_bigrams'])}

    Average Words per Sentence: {processed_text['words_per_sentence']:.2f}
    """

    # Make the API call to Claude 3.5
    try:
        response = anthropic.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=1024,
            messages=[
                {"role": "user", "content": f"Here's some processed text from an audio transcript:\n\n{context}\n\nBased on this information, please answer the following question: {question}"}
            ]
        )
        return response.content[0].text
    except Exception as e:
        return f"An error occurred while querying Claude: {str(e)}"

def detect_inference(processed_text):
    # Placeholder function for inference detection
    # This would need to be implemented based on specific criteria for your use case
    return []

def main():
    verify_cuda()
    
    # Get the directory of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Prompt for the audio file name (not the full path)
    audio_file_name = input("Enter the name of your audio file (e.g., MLKDream_64kb.mp3): ")
    
    # Construct the full path
    audio_file_path = os.path.join(script_dir, audio_file_name)
    
    if not os.path.exists(audio_file_path):
        logging.error(f"File not found: {audio_file_path}")
        return

    logging.info(f"Attempting to transcribe: {audio_file_path}")
    transcribed_text = transcribe_audio(audio_file_path)
    
    if transcribed_text:
        logging.info("Transcription successful.")
        processed_text = process_text(transcribed_text)
        
        # Print some basic analysis
        print("\nBasic Analysis:")
        print(f"Number of words: {len(processed_text['tokens'])}")
        print(f"Number of unique words: {len(processed_text['word_frequency'])}")
        print(f"Named entities found: {', '.join(processed_text['named_entities'])}")
        print(f"Average words per sentence: {processed_text['words_per_sentence']:.2f}")
        print("Most common words:", ', '.join(f"{word}({count})" for word, count in processed_text['most_common_words']))
        print("Common bigrams:", ', '.join(f"{' '.join(bigram)}({count})" for bigram, count in processed_text['common_bigrams']))
        
        while True:
            user_question = input("\nAsk a question about the audio content (or type 'quit' to exit): ")
            if user_question.lower() == 'quit':
                break
            
            answer = analyze_with_llm(processed_text, user_question)
            print("\nClaude's answer:", answer)
            
            inferences = detect_inference(processed_text)
            if inferences:
                print("\nPotential inferences or indirect references detected:")
                for inference in inferences:
                    print(f"- {inference}")
    else:
        logging.error("Transcription failed. Please check the audio file and try again.")

if __name__ == "__main__":
    main()