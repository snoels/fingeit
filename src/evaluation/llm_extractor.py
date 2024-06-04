import re

def open_ai_request(client, message):
  response = client.chat.completions.create(
    messages=message,
      model="gpt-3.5-turbo",    
      temperature=0,        
      max_tokens=2048         
  )
  return response.choices[0].message.content


def relationship_present(pred, rels):

    rels_present = []

    for r in rels:
        if r.lower() in pred.lower():
            rels_present.append(r)
    
    if len(rels_present) > 1:
        return []
    elif len(rels_present) == 0:
        return []
    else:
        return rels_present


def generate_classification_message(instruction: str, prediction: str):

    scale_string = re.search('{(.*)}', instruction).group(1)
    answer_scale = scale_string.split('/')

    scale_string = ', '.join(f"'{w}'" for w in answer_scale[:-1])
    scale_string += f", and '{answer_scale[-1]}'"
    system = (
        f"You are a helpful assistant specialized in extracting the label of a message:"
        f" the possible labels are {scale_string}. If none of the labels apply reply with 'unknown'."
    )

    scale_string = ', '.join(answer_scale + ["unknown"])
    instruction = f"Determine the label of the message. Options: {scale_string}\nno other options may be given"

    message = [
        {
            "role": "system",
            "content": system,
        },
        {
            "role": "user",
            "content": f"{system}\n\n### Instruction: {instruction}\n\n### Input: {prediction}\n\n### Response:",
        },
    ]
    return message

def generate_message_ner(instruction: str, prediction: str):

    system = (
        "You are a helpful assistant specialized in formatting a message in the right format:"
        " answer the question by answering in the following format: [entity] is een [type], [entity] is a [type], ..."
    )

    instruction = "Please extract entities and their types from the input sentence, entity types must be chosen from {persoon/organisatie/locatie}."

    message = [
        {
            "role": "system",
            "content": system,
        },
        {
            "role": "user",
            "content": f"{system}\n\n### Instruction: {instruction}\n\n### Input: {prediction}\n\n### Response:",
        },
    ]
    return message

def generate_message_finred(instruction: str, prediction: str, answer_scale):

    scale_string = ', '.join(f"'{w}'" for w in answer_scale[:-1])
    scale_string += f", and '{answer_scale[-1]}'"
    system = (
        f"You are a helpful assistant specialized in extracting the label of a message:"
        f" the possible labels are {scale_string}. If none of the labels apply reply with 'unknown'."
    )

    scale_string = ', '.join(answer_scale + ["unknown"])
    instruction = f"Determine the label of the message. Options: {scale_string}\nno other options may be given"

    message = [
        {
            "role": "system",
            "content": system,
        },
        {
            "role": "user",
            "content": f"{system}\n\n### Instruction: {instruction}\n\n### Input: {prediction}\n\n### Response:",
        },
    ]
    return message

def generate_message_convfinqa(instruction: str, prediction: str):

    system = (
        "You are a helpful assistant specialized in extracting the requested numerical value out of a message:"
        " answer the question by just returning the requested numerical value. If a calculation is provided, ignore this information."
    )

    instruction = "Please extract the numerical value that resides within a message. If this is not the case, answer with 'unknown.'"
    
    message = [
        {
            "role": "system",
            "content": system,
        },
        {
            "role": "user",
            "content": f"{system}\n\n### Instruction: {instruction}\n\n### Input: {prediction}\n\n### Response:",
        },
    ]
    return message

def extracted_answers(df, client):

  extracted_answers = []

  for _, row in df.iterrows():
    extracted_answers.append(open_ai_request(client, generate_classification_message(row['instruction'], row['prediction'])))

  return extracted_answers


def extracted_answers_ner(df, client):

  extracted_answers = []

  for _, row in df.iterrows():
    extracted_answers.append(open_ai_request(client, generate_message_ner(row['instruction'], row['prediction'])))

  return extracted_answers

def extracted_answers_finred(df, client):
    
    relations = [
    "product/materiaal geproduceerd",
    "fabrikant",
    "verdeeld door",
    "industrie",
    "positie bekleed",
    "originele omroep",
    "bezeten door",
    "opgericht door",
    "distributieformaat",
    "hoofdkantoorlocatie",
    "effectenbeurs",
    "valuta",
    "moederorganisatie",
    "chief executive officer",
    "directeur/manager",
    "eigenaar van",
    "operator",
    "lid van",
    "werkgever",
    "voorzitter",
    "platform",
    "dochteronderneming",
    "rechtsvorm",
    "uitgever",
    "ontwikkelaar",
    "merk",
    "bedrijfsdivisie",
    "locatie van ontstaan",
    "maker",
    ]

    extracted_answers = []

    exact_extractions = 0

    for _, row in df.iterrows():

        rels_present = relationship_present(row['prediction'], relations)

        if len(rels_present) == 1:
            extracted_answers.append(rels_present[0])
            exact_extractions += 1
        else:
            llm_extraction = open_ai_request(client, generate_message_finred(row['instruction'], row['prediction'],relations))

            rels_present = relationship_present(llm_extraction, relations)

            if len(rels_present) == 1:
                extracted_answers.append(rels_present[0])
            else:
                extracted_answers.append('unknown')

    return extracted_answers


def extracted_answers_convfinqa(df, client):

  extracted_answers = []

  for _, row in df.iterrows():
    extracted_answers.append(open_ai_request(client, generate_message_convfinqa(row['instruction'], row['prediction'])))

  return extracted_answers


