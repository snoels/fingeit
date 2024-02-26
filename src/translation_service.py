import asyncio
import aiohttp
import ssl
import certifi

async def call_chatgpt_async(session, config, target: str):
    payload = {
        'model': config.get('DEFAULT', 'model'),
        'messages': [
            {"role": "system", "content": config.get('DEFAULT', 'system_prompt')},
            {"role": "user", "content": config.get('DEFAULT', 'prompt') + '\n\n' + target}
        ],
        "max_tokens": config.getint('DEFAULT', 'max_tokens'),
        "temperature" : config.getfloat('DEFAULT', 'temperature')
    }
    try:
        async with session.post(
            url='https://api.openai.com/v1/chat/completions',
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {config.get('DEFAULT', 'openai_secret_key')}"},
            json=payload,
            ssl=ssl.create_default_context(cafile=certifi.where())
        ) as response:
            response = await response.json()
        if "error" in response:
            print(f"OpenAI request failed with error {response['error']}")
        return response['choices'][0]['message']['content']
    except:
        print("Request failed.")

async def call_chatgpt_bulk(prompts, config):
    async with aiohttp.ClientSession() as session:
        responses = await asyncio.gather(*[call_chatgpt_async(session, config, prompt) for prompt in prompts])
    return responses

def alpaca_prompt(instruction, input, response):
    return f"""### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{response}"""

def alpaca_empty_input_prompt(instruction, response):
    return f"""### Instruction:\n{instruction}\n\n### Response:\n{response}"""
