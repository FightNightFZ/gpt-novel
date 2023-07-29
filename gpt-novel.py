import openai
import os
import pickle
import requests
import base64
import tempfile
import os
import shutil
from ebooklib import epub
import time
import datetime
import gradio as gr
from retrying import retry
import nltk
import logging

logging.basicConfig(filename='novel_generation.log', level=logging.INFO)

def break_into_paragraphs(text, sentences_per_paragraph=5):
    sentences = nltk.sent_tokenize(text)
    paragraphs = [' '.join(sentences[i:i+sentences_per_paragraph]) 
                  for i in range(0, len(sentences), sentences_per_paragraph)]
    formatted_text = "<p>" + "</p><p>".join(paragraphs) + "</p>"
    return formatted_text

def estimate_tokens(text):
    return int(len(text.encode('utf-8'))) // 4
    
def retry_if_api_error(exception):
    return isinstance(exception, openai.error.APIError)

@retry(retry_on_exception=retry_if_api_error, stop_max_attempt_number=7, wait_exponential_multiplier=1000, wait_exponential_max=10000)

def openai_api_call_with_rate_limit_handling(model, messages):
    max_tokens = 8192 if model == "gpt-4" else 4096
    max_tokens_for_completion = int(max_tokens - 100)  # reserve some tokens for the messages

    total_token_count = sum(int(estimate_tokens(message['content'])) for message in messages)
    while total_token_count > max_tokens:
        messages.pop(0)
        total_token_count = sum(int(estimate_tokens(message['content'])) for message in messages)

    response = None
    while True:
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                n=1,
                max_tokens=min(int(max_tokens_for_completion), int(max_tokens - total_token_count)),
                temperature=0.6,
            )
            break
        except Exception as error:
            if "rate limit" in str(error).lower():
                print("Rate limit hit, sleeping for a minute...")
                time.sleep(60)
            else:
                raise

    return response

def print_step_costs(response, model):
    if model == "gpt-4":
        input_per_token = 0.00003
        output_per_token = 0.00006
    elif model == "gpt-3.5-turbo-16k":
        input_per_token = 0.000003
        output_per_token = 0.000004
    elif model == "gpt-3.5-turbo":
        input_per_token = 0.0000015
        output_per_token = 0.000002

    input_cost = response['usage']['prompt_tokens'] * input_per_token
    output_cost = response['usage']['completion_tokens'] * output_per_token
    total_cost = input_cost + output_cost

    print('step cost:', total_cost)

def generate_detailed_outline(prompt, genre, author_style, gpt_version):
    logging.info("Generating detailed outline...")
    messages = [
        {"role": "system", "content": f"You are a highly skilled AI assistant, specifically trained in the art of creative writing. Please write in a style similar to {author_style}. Your task is to generate a detailed outline for a novel. The genre of the novel is {genre.lower()}."},
        {"role": "user", "content": f"Your task is to generate a detailed outline for a novel. The novel should be in the {genre.lower()} genre and should be inspired by the following prompt: {prompt}."}
    ]
    response = openai_api_call_with_rate_limit_handling(gpt_version, messages)
    print_step_costs(response, gpt_version)
    response_content = response['choices'][0]['message']['content']
    logging.info("Response content: %s", response_content)
    return response_content

def select_most_engaging(plots, author_style, gpt_version):
    logging.info("Selecting most engaging plot...")
    messages = [
        {"role": "system", "content": f"You are an AI assistant with expertise in crafting fantastic novel plots. Please write in a style similar to {author_style}."},
        {"role": "user", "content": f"Below are a number of potential plots for a new novel: {plots}\n\nYour task is to craft the final plot for our novel."}
    ]
    response = openai_api_call_with_rate_limit_handling(gpt_version, messages)
    print_step_costs(response, gpt_version)
    return response['choices'][0]['message']['content']

def improve_plot(plot, author_style, gpt_version):
    logging.info("Improving plot...")
    messages = [
        {"role": "system", "content": f"You are an AI assistant with expertise in improving and refining story plots. Please write in a style similar to {author_style}."},
        {"role": "user", "content": f"Your task is to refine and enhance the following plot: {plot}."}
    ]
    response = openai_api_call_with_rate_limit_handling(gpt_version, messages)
    print_step_costs(response, gpt_version)
    return response['choices'][0]['message']['content']

def get_title(plot, gpt_version):
    logging.info("Generating title...")
    messages = [
        {"role": "system", "content": "You are an AI assistant with expertise in creative writing."},
        {"role": "user", "content": f"Given the plot: {plot}, your task is to come up with a unique and fitting title for the book. Please avoid common or clichéd phrases."}
    ]
    response = openai_api_call_with_rate_limit_handling(gpt_version, messages)
    title = response['choices'][0]['message']['content']
    return title.replace('Title: "', '').replace('": Chapter 1', '')

def write_first_chapter(storyline, first_chapter_title, writing_style, author_style, gpt_version):
    print("Writing first chapter...")  # New print statement
    messages = [
        {"role": "system", "content": f"You are an AI assistant with expertise in creative writing. Please write in a style similar to {author_style}."},
        {"role": "user", "content": f"Given the high-level plot: {storyline}, your task is to write the first chapter of a novel, titled `{first_chapter_title}`. Follow the writing style described here: `{writing_style}`."},
        {"role": "user", "content": "Please avoid generating a summary or ending paragraph at the end of the chapter."}
    ]
    response = openai_api_call_with_rate_limit_handling(gpt_version, messages)
    print_step_costs(response, gpt_version)
    return response['choices'][0]['message']['content']

def write_chapter(plot, previous_chapters, chapter_title, writing_style, author_style, book_title, gpt_version):
    print("Writing chapter...")
    print("Calling write_chapter with the following inputs:")
    print(f"Plot: {plot}")
    print(f"Previous chapters: {previous_chapters}")
    print(f"Chapter title: {chapter_title}")
    print(f"Writing style: {writing_style}")
    print(f"Author style: {author_style}")
    print(f"Book title: {book_title}")

    if previous_chapters:  
        messages = [
            {"role": "system", "content": f"You are an AI assistant with expertise in creative writing. Please write in a style similar to {author_style}."},
            {"role": "user", "content": f"Given the plot: {plot} and the previous chapters: {previous_chapters}, your task is to write the next chapter of a novel titled '{book_title}'. Consider the plan for this chapter: {chapter_title}."},
            {"role": "user", "content": "Please avoid generating a summary or ending paragraph at the end of the chapter."}
        ]
    else:
        messages = [
            {"role": "system", "content": f"You are an AI assistant with expertise in creative writing. Please write in a style similar to {author_style}."},
            {"role": "user", "content": f"Given the plot: {plot}, your task is to write the first chapter of a novel titled '{book_title}'. Consider the plan for this chapter: {chapter_title}."},
            {"role": "user", "content": "Please avoid generating a summary or ending paragraph at the end of the chapter."}
        ]

    response = openai_api_call_with_rate_limit_handling(gpt_version, messages)
    print_step_costs(response, gpt_version)
    chapter_content = response['choices'][0]['message']['content']
    # Split chapter content by newline
    lines = chapter_content.split('\n')
    # If the first line doesn't end with a period, discard it
    if not lines[0].endswith('.'):
        lines.pop(0)
    # Join the remaining lines back together
    chapter_content = '\n'.join(lines)
    return chapter_content

def generate_cover_description(plot, gpt_version):
    print("Generating cover description...")
    
    # Truncate plot at the end of a sentence if it's too long
    if len(plot) > 2000:
        truncated_plot = ""
        sentences = nltk.sent_tokenize(plot)
        for sentence in sentences:
            if len(truncated_plot + sentence) <= 2000:
                truncated_plot += sentence
            else:
                break
        plot = truncated_plot

    messages = [
        {"role": "system", "content": "You are an AI assistant with expertise in creative writing and graphic design."},
        {"role": "user", "content": f"Given the plot: {plot}, generate a detailed description for the cover of the novel."}
    ]
    response = openai_api_call_with_rate_limit_handling(gpt_version, messages)
    print_step_costs(response, gpt_version)
    description = response['choices'][0]['message']['content']
    
    return description

def create_cover_image(plot, stability_api_key, gpt_version, engine_id="stable-diffusion-xl-beta-v2-2-2"):  
    print("Creating cover image...")
    cover_description = generate_cover_description(plot, gpt_version)  
    
    # Ensure cover_description is not empty or whitespace
    if not cover_description.strip():
        raise ValueError("Cover description cannot be empty or consist only of whitespace.")

    # Ensure cover_description is within the required length
    if len(cover_description) > 2000:
        print("Cover description length is more than 2000 characters. Adjusting...")
        cover_description = cover_description[:2000]

    api_host = os.getenv('API_HOST', 'https://api.stability.ai')

    if stability_api_key is None:
        raise Exception("Missing Stability API key.")

    response = requests.post(
        f"{api_host}/v1/generation/{engine_id}/text-to-image",
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {stability_api_key}"
        },
        json={
            "text_prompts": [
                {
                    "text": cover_description
                }
            ],
            "cfg_scale": 5,  # Lowering cfg_scale to 5
            "clip_guidance_preset": "FAST_BLUE",
            "height": 768,
            "width": 512,
            "samples": 1,
            "steps": 50,  # Increasing steps to 50
        },
    )

    print("Full response:", response.json())  

    if response.status_code != 200:
        print("API request failed with status code:", response.status_code)
        print("Full response:", response.json())
        raise Exception("Non-200 response: " + str(response.text))

    response_json = response.json()
    artifacts = response_json.get("artifacts")
    if artifacts is None or not artifacts:
        print("No artifacts in response.")
        print("Full response:", response_json)
        raise Exception("No artifacts in response.")

    # Get the first artifact
    artifact = artifacts[0]
    image_data = artifact.get("base64")
    if image_data is None:
        print("Image data not found in response.")
        print("Full response:", response_json)
        raise Exception("Image data not found in response.")

    print(f"Image data: {image_data}")  # Debugging print statement

    image_bytes = base64.b64decode(image_data)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp:
        temp.write(image_bytes)
        temp_image_path = temp.name

    # Move the image to a permanent location
    permanent_image_path = os.path.join(os.getcwd(), 'cover_image.png')
    shutil.move(temp_image_path, permanent_image_path)

    return permanent_image_path

def create_epub(title, author, chapters, cover_image_path):
    print("Creating ePub...")  
    book = epub.EpubBook()
    book.set_identifier('id123456')
    book.set_title(title)
    book.set_language('en')
    book.add_author(author)

    if cover_image_path:
        try:
            with open(cover_image_path, 'rb') as cover_file:
                cover_image = cover_file.read()
            book.set_cover("cover.png", cover_image)
        except Exception as e:
            logging.exception("Failed to add cover image: ")
            return str(e)

    epub_chapters = []
    for i, chapter_dict in enumerate(chapters, start=1):
        for chapter_title, chapter_content in chapter_dict.items():
            chapter_file_name = f'chapter_{i}.xhtml'
            epub_chapter = epub.EpubHtml(title=chapter_title, file_name=chapter_file_name, lang='en')
            formatted_content = break_into_paragraphs(chapter_content)
            epub_chapter.content = f'<h1>{chapter_title}</h1>{formatted_content}'
            book.add_item(epub_chapter)
            epub_chapters.append(epub_chapter)

    book.toc = (epub_chapters)
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())

    style = '''
    @namespace epub "http://www.idpf.org/2007/ops";
    body {
        font-family: Cambria, Liberation Serif, serif;
    }
    h1 {
        text-align: left;
        text-transform: uppercase;
        font-weight: 200;
    }
    '''
    nav_css = epub.EpubItem(uid="style_nav", file_name="style/nav.css", media_type="text/css", content=style)
    book.add_item(nav_css)
    book.spine = ['nav'] + epub_chapters

    now = datetime.datetime.now()
    now_str = now.strftime("%Y%m%d%H%M%S")
    safe_title = ''.join(c if c.isalnum() else '_' for c in title)
    filename = f"{safe_title}_{now_str}.epub"
    epub_path = os.path.join(os.getcwd(), filename)
    epub.write_epub(epub_path, book)

    return epub_path

def text_to_speech(text, tts_api_key):
    base_url = "https://api.voicerss.org/"
    params = {
        "key": tts_api_key,
        "src": text,
        "hl": "en-us"
    }

    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()  # Raises stored HTTPError, if one occurred.
        print('TTS API request successful')  
    except (requests.exceptions.RequestException, requests.exceptions.HTTPError) as err:
        print(f"An error occurred while making the text-to-speech request: {err}")
        return None

    print(f"Response content type: {response.headers['content-type']}")  # Check response content type
    audio_content = response.content

    if not audio_content:
        print("No audio content in the response.")
        return None

    audio_file_path = os.path.join(os.getcwd(), "novel_audio.mp3")
    try:
        with open(audio_file_path, 'wb') as audio_file:
            audio_file.write(audio_content)
        print(f'Audio file written to {audio_file_path}')  
    except IOError as err:
        print(f"An error occurred while writing the audio file: {err}")
        return None

    return audio_file_path

def generate_novel(genre, prompt, writing_style, num_chapters, feedback_after_each_chapter, openai_key, stability_key, author_style, feedback, gpt_version, save_filename, tts_api_key=None):
    print(f'TTS API Key: {tts_api_key}') 
    logging.info("Generating novel...")  

    # Check if the keys are provided
    if not openai_key:
        logging.error("Missing OpenAI key.")
        return "Error: Missing OpenAI key.", None, None
    if not stability_key:
        logging.error("Missing Stability key.")
        return "Error: Missing Stability key.", None, None
        
    openai.api_key = openai_key
    feedback_after_each_chapter = feedback_after_each_chapter == "Yes"
    
    chapters_string = ''  

    try:
        detailed_outline = generate_detailed_outline(prompt, genre, author_style, gpt_version)
        engaging_plot = select_most_engaging(detailed_outline, author_style, gpt_version)
        improved_plot = improve_plot(engaging_plot, author_style, gpt_version)
        title = get_title(improved_plot, gpt_version).replace('Title: "', '').replace('": Chapter 1', '')

        chapters = []
        for i in range(1, int(num_chapters) + 1):  
            if i == 1:
                first_chapter_content = write_first_chapter(improved_plot, "Chapter 1", writing_style, author_style, gpt_version)
                first_chapter_title = get_title(first_chapter_content, gpt_version).replace('Title: "', '').replace('": Chapter 1', '')
                first_chapter = {first_chapter_title: first_chapter_content}
                chapters.append(first_chapter)
            else:
                chapter_texts = [f"{title}: {content}" for chapter in chapters for title, content in chapter.items()]
                plot_for_chapter = f"{improved_plot}\n\nSo far, {'. '.join(chapter_texts)}"
                chapter_title = f"Chapter {i}"

                if feedback_after_each_chapter:
                    feedback = input("Your feedback: ")
                    chapter_texts = [content for chapter in chapters for content in chapter.values()]
                    chapter_content = write_chapter(plot_for_chapter + "\n\n" + feedback, '. '.join(chapter_texts), chapter_title, writing_style, author_style, title, gpt_version)
                else:
                    chapter_texts = [content for chapter in chapters for content in chapter.values()]
                    chapter_content = write_chapter(plot_for_chapter, '. '.join(chapter_texts), chapter_title, writing_style, author_style, title, gpt_version)

                chapter_title = get_title(chapter_content, gpt_version).replace('Title: "', '').replace('": Chapter 1', '')
                chapter = {chapter_title: chapter_content}
                chapters.append(chapter)

            with open(f'{save_filename}.pickle', 'wb') as f:
                pickle.dump(chapters, f)

        cover_image_path = create_cover_image(improved_plot, stability_key, gpt_version)
        epub_path = create_epub(title, "AI Author", chapters, cover_image_path)

        # Convert chapters to a single string
        chapters_string = '\n'.join([f"{title}: {content}" for chapter in chapters for title, content in chapter.items()])

        # Add TTS functionality
        audio_file_path = None
        if tts_api_key:
            try:
                audio_file_path = text_to_speech(chapters_string, tts_api_key)
                print(f"Audio file saved at {audio_file_path}")
            except Exception as e:
                print(f"An error occurred while converting text to speech: {e}")

        return epub_path, chapters_string, audio_file_path
    except Exception as e:
        print(f"An error occurred while generating the novel: {e}")
        return None, None, None
        
# Define Gradio interface options
    genre_options = ["Romance", "Science Fiction", "Fantasy", "Horror", "Mystery", "Thriller", "Western", "Historical", "Literary"]
    num_chapters_options = [str(x) for x in range(1, 21)]
    feedback_options = ["Yes", "No"]
    
    genre = gr.Dropdown(genre_options, label="Genre")
    prompt = gr.Textbox(lines=2, placeholder="Plot prompt (optional)", label="Prompt")
    writing_style = gr.Textbox(lines=5, placeholder="Description of the writing style (optional)", label="Writing Style")
    num_chapters = gr.Dropdown(num_chapters_options, label="Number of Chapters")
    feedback_after_each_chapter = gr.Dropdown(feedback_options, label="Would you like to give feedback after each chapter?")
    openai_key = gr.Text(type="password", label="OpenAI API Key")
    stability_key = gr.Text(type="password", label="Stability API Key")
    author_style = gr.Textbox(lines=2, placeholder="Author whose style the AI should emulate (optional)", label="Author Style")
    feedback = gr.Textbox(lines=5, placeholder="Your feedback for the AI (optional)", label="Feedback")
    gpt_version = gr.Dropdown(['gpt-4', 'gpt-3.5-turbo-16k', 'gpt-3.5-turbo'], label="GPT Version")
    save_filename = gr.Textbox(label="Save Filename")
    tts_api_key = gr.Text(type="password", label="TTS API Key")

# Define Gradio Interface
    demo = gr.Interface(generate_novel, 
                    [genre, prompt, writing_style, num_chapters, feedback_after_each_chapter, openai_key, stability_key, author_style, feedback, gpt_version, save_filename, tts_api_key], 
                    "text", 
                    title="AI Novel Generator")

    if __name__ == "__main__":
        logging.info("Starting main...")
        demo.launch()