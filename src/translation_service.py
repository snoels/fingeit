import asyncio
import ssl

import aiohttp
import certifi


async def call_chatgpt_async(session, config, target: str):
    payload = {
        "model": config.get("TRANSLATE", "model"),
        "messages": [
            {"role": "system", "content": config.get("TRANSLATE", "system_prompt")},
            {
                "role": "user",
                "content": config.get("TRANSLATE", "prompt") + "\n\n" + target,
            },
        ],
        "max_tokens": config.getint("TRANSLATE", "max_tokens"),
        "temperature": config.getfloat("TRANSLATE", "temperature"),
    }
    try:
        async with session.post(
            url="https://api.openai.com/v1/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {config.get('TRANSLATE', 'openai_secret_key')}",
            },
            json=payload,
            ssl=ssl.create_default_context(cafile=certifi.where()),
        ) as response:
            response = await response.json()
        if "error" in response:
            print(f"OpenAI request failed with error {response['error']}")
        return response["choices"][0]["message"]["content"]
    except:
        print("Request failed.")


async def call_chatgpt_bulk(prompts, config):
    async with aiohttp.ClientSession() as session:
        responses = await asyncio.gather(
            *[call_chatgpt_async(session, config, prompt) for prompt in prompts]
        )
    return responses


def alpaca_prompt(instruction, input, response):
    return f"""### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{response}"""


def alpaca_empty_input_prompt(instruction, response):
    return f"""### Instruction:\n{instruction}\n\n### Response:\n{response}"""
