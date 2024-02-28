import asyncio
import ssl

import aiohttp
import certifi
import pandas as pd
import requests
from tqdm.auto import tqdm


def create_payload(target, config):
    return {
        "model": config.get("model"),
        "messages": [
            {"role": "system", "content": config.get("system_prompt")},
            {
                "role": "user",
                "content": config.get("prompt") + "\n\n" + target,
            },
        ],
        "max_tokens": config.getint("max_tokens"),
        "temperature": config.getfloat("temperature"),
    }


async def call_chatgpt_async(session, config, target: str) -> str | None:
    """Will call the chatgpt model asynchronously and return the response.

    will return the translation if successful, else None
    """
    try:
        async with session.post(
            url="https://api.openai.com/v1/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {config.get('openai_secret_key')}",
            },
            json=create_payload(target, config),
            ssl=ssl.create_default_context(cafile=certifi.where()),
        ) as response:
            response = await asyncio.wait_for(response.json(), timeout=60)

        if "error" in response:
            print(f"OpenAI request failed with error {response['error']}")
        return response["choices"][0]["message"]["content"]
    except asyncio.TimeoutError:
        print("The request has timed out.")
    except Exception as e:
        print("Request failed: ", str(e))
    return None


async def call_chatgpt_bulk(prompts, config):
    chunk_size = config.getint("download_chunk_size")
    async with aiohttp.ClientSession() as session:
        responses = []
        with tqdm(total=len(prompts), desc="Translating") as pbar:
            for i in range(0, len(prompts), chunk_size):
                end = (
                    i + chunk_size if (i + chunk_size < len(prompts)) else len(prompts)
                )
                prompts_chunk = prompts[i:end]
                chunk_responses = await asyncio.gather(
                    *[
                        call_chatgpt_async(session, config, prompt)
                        for prompt in prompts_chunk
                    ]
                )
                responses += chunk_responses
                pbar.update(len(chunk_responses))
    return responses


def call_chatgpt_sync(config, target: str):
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
        response = requests.post(
            url="https://api.openai.com/v1/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {config.get('TRANSLATE', 'openai_secret_key')}",
            },
            json=payload,
            verify=certifi.where(),
        )
        response = response.json()
        if "error" in response:
            print(f"OpenAI request failed with error {response['error']}")
        return response["choices"][0]["message"]["content"]
    except:
        print("Request failed.")


def translate_and_add_response(dataset, config):
    """Translates dataset from prompts and adds response to dataset."""
    for dataset_keys in dataset.keys():
        pd_df = pd.DataFrame(dataset[dataset_keys])
        prompts = list(pd_df["prompt"])

        responses = asyncio.run(call_chatgpt_bulk(prompts, config))

        dataset_with_translation = dataset[dataset_keys].add_column(
            "translation", responses
        )
        dataset[dataset_keys] = dataset_with_translation

    return dataset
