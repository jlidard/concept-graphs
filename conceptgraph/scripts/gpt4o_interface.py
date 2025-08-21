import logging
import os
import re
from textwrap import wrap
import base64

import httpx
from openai import OpenAI

# from predictive_brickwork.vlm.utils import encode_image

# Set the logging level for httpx to WARNING or higher (disables INFO messages from OpenAI)
logging.getLogger("httpx").setLevel(logging.WARNING)




def encode_image(image_path):
    # Open the image file and encode it as a base64 string
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def strip_format(formatted_string, format_type=None):
    """
    Strips a formatted string like ```python<string>``` or ```json<string>```
    and returns the string inside the angle brackets. If the format type is not python,
    it also strips any comments starting with two slashes (//).

    Args:
        formatted_string (str): The formatted string to process.
        format_type (str): The format type of the string (e.g., "python", "json").

    Returns:
        str: The processed string inside the angle brackets, or None if not found.
    """
    if format_type:
        match = re.search(rf"```(?:{format_type})?\n(.*?)\n```", formatted_string, re.DOTALL)
    else:
        match = re.search(r"```.*?\n(.*?)\n```", formatted_string, re.DOTALL)
    if match:
        content = match.group(1)
        # content = re.sub(r'//.*', '', content)
        return content
    return formatted_string


class GPT4OInterface:
    def __init__(
        self,
        save_file=None,
        system_prompt=None,
        model="gpt-4o",
        max_tokens=3000,
        return_logprobs=False,
        extraction_format=None,
        eliminate_whitespace_file_inputs=False,
        update_history=True,
    ):
        self.api_key = os.getenv("OPENAI_GENERATIVE_CONSTRUCTION_KEY")
        self.system_prompt = system_prompt
        if self.system_prompt is None:
            self.system_prompt = "You are a helpful assistant that will provide guidance to a user. "
        self.model = model
        self.client = OpenAI(api_key=self.api_key)
        self.save_file = save_file
        self.max_tokens = max_tokens
        self.return_logprobs = return_logprobs
        self.extraction_format = extraction_format
        self.eliminate_whitespace_file_inputs = eliminate_whitespace_file_inputs
        self.update_history = update_history  # turn off if updating externally

    def generate_payload(
        self, prompt, image_path=None, icl_paths=None, code_paths=None, additional_images=None, chat_history=None
    ):
        # Initialize user instruction
        prompt = "User instruction: " + prompt

        # Initialize content
        content_list = [
            {"type": "text", "text": prompt},
        ]

        # Add the images if available
        for images in [image_path, additional_images]:
            if images is not None:
                if not isinstance(images, list):
                    images = [images]
                for image in images:
                    base64_image = encode_image(image)
                    image_payload = {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    }
                    content_list.append(image_payload)

        content_message_user = {"role": "user", "content": content_list}
        content_message_system = {"role": "system", "content": self.system_prompt}

        content_message_code = None
        if code_paths is not None:
            code_str = ""
            for cp in code_paths:
                with open(cp, "r") as file:
                    example_code = file.read()
                    if self.eliminate_whitespace_file_inputs:
                        example_code = re.sub(r"\s+", "", example_code)
                    code_str += "\n{}".format(example_code)
            code_str = f"API:\n{code_str}"
            content_message_code = {"role": "user", "content": code_str}

        if chat_history and self.update_history:
            # If previous payload exists, add the new content to it
            chat_history.append(content_message_user)
            if content_message_code:
                chat_history.append(content_message_code)
            return chat_history

        payload = [content_message_system, content_message_user]
        if content_message_code:
            payload.append(content_message_code)

        if icl_paths is not None:
            icl_str = ""
            for idx, icl in enumerate(icl_paths, start=1):
                with open(icl, "r") as file:
                    example_icl = file.read()
                    if self.eliminate_whitespace_file_inputs:
                        example_icl = re.sub(r"\s+", "", example_icl)
                    icl_str += "\nExample {}:\n{}".format(idx, example_icl)
            icl_str = f"In-context examples:\n{icl_str}"
            payload.append({"role": "user", "content": icl_str})

        return payload

    def generate_completion(
        self,
        prompt,
        image_path=None,
        icl_paths=None,
        code_paths=None,
        additional_images=None,
        temperature=0.2,
        chat_history=None,
    ):
        payload = self.generate_payload(prompt, image_path, icl_paths, code_paths, additional_images, chat_history)

        params = {
            "model": self.model,
            "messages": payload,
            "temperature": temperature,
        }

        if self.return_logprobs:
            params["logprobs"] = True
            params["top_logprobs"] = 20

        if self.max_tokens:
            if self.model == "o1-preview" or self.model == "o1" or self.model == "o1-mini" or self.model == "o3-mini":
                params["max_completion_tokens"] = self.max_tokens
            else:
                params["max_tokens"] = self.max_tokens

        completion = self.client.chat.completions.create(**params)

        return completion, payload

    def unpack_completion(self, completion):
        content = completion.choices[0].message.content

        key_logprobs = None
        if self.return_logprobs:
            logprobs = completion.choices[0].logprobs.content[0].top_logprobs
            keys = ["A", "B", "C", "D", "E"]

            # take the max logprobs out of all variations of the toke (e.g., 'A', ' A', '"A')
            key_logprobs = {k: [] for k in keys}
            for logprob in logprobs:
                if logprob.token.strip().upper() in keys:
                    key = logprob.token.strip().upper()
                    key_logprobs[key].append(logprob.logprob)

            for key in keys:
                if len(key_logprobs[key]) > 0:
                    key_logprobs[key] = max(key_logprobs[key])
                else:
                    key_logprobs[key] = -30.0
        return content, key_logprobs

    def maybe_save_to_file(self, completion):
        if self.save_file is not None:
            completion_stripped = strip_format(completion, format_type=self.extraction_format)

            with open(self.save_file, "w") as f:
                if completion_stripped:
                    f.write(completion_stripped)
                else:
                    f.write(completion)
                    print("Warning: The stripped completion is empty. Saving the original completion instead.")

    def query(
        self,
        prompt,
        image_path=None,
        icl_paths=None,
        code_paths=None,
        additional_images=None,
        temperature=0.2,
        chat_history=None,
    ):
        completion, payload = self.generate_completion(
            prompt, image_path, icl_paths, code_paths, additional_images, temperature, chat_history
        )
        response, key_logprobs = self.unpack_completion(completion)
        self.maybe_save_to_file(response)

        # update the payload with the response
        result_metadata = {"role": "assistant", "content": response}
        payload.append(result_metadata)

        return response, key_logprobs, payload


def annotate_images_with_letters(directory_path):
    import os

    from PIL import Image, ImageDraw, ImageFont

    # Ensure the output directory exists
    annotated_dir = os.path.join(directory_path, "annotated")
    os.makedirs(annotated_dir, exist_ok=True)

    # Define the letters and set up font properties
    letters = ["A", "B", "C", "D", "E"]

    # Loop through the images in the directory
    for i, file_name in enumerate(sorted(os.listdir(directory_path))):
        # Process only image files
        if file_name.lower().endswith((".png", ".jpg", ".jpeg")):
            image_path = os.path.join(directory_path, file_name)
            image = Image.open(image_path)
            draw = ImageDraw.Draw(image)

            # Get image dimensions
            width, height = image.size

            font = ImageFont.truetype("arial.ttf", width // 10)  # Adjust the font size as needed

            # Set the position for the letter (centered)
            text = letters[i % len(letters)]  # Cycle through A to E
            text_size = draw.textlength(text, font=font)
            position = 0, 0

            # Draw the letter in bright red
            draw.text(position, text, (255, 0, 0), font=font)

            # Save the annotated image
            output_path = os.path.join(annotated_dir, f"annotated_{file_name}")
            image.save(output_path)