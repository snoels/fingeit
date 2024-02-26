from src.evaluation.answer_extractor import relationship_present, open_ai_request, generate_message_finred

def generate_message_ner(instruction: str, prediction: str):

    system = (
        "You are a helpful assistant specialized in formatting a message in the right format:"
        " answer the question by answering in the following format: [entity] is a [type], [entity] is a [type], ..."
    )

    instruction = "Please extract entities and their types from the input sentence, entity types must be chosen from {person/organization/location}."

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

def extracted_answers_ner(df, client):

  extracted_answers = []

  for _, row in df.iterrows():
    extracted_answers.append(open_ai_request(client, generate_message_ner(row['instruction'], row['prediction'])))

  return extracted_answers

def extracted_answers_finred(df, client):
    
    relations = [
    "product_or_material_produced",
    "manufacturer",
    "distributed_by",
    "industry",
    "position_held",
    "original_broadcaster",
    "owned_by",
    "founded_by",
    "distribution_format",
    "headquarters_location",
    "stock_exchange",
    "currency",
    "parent_organization",
    "chief_executive_officer",
    "director_/_manager",
    "owner_of",
    "operator",
    "member_of",
    "employer",
    "chairperson",
    "platform",
    "subsidiary",
    "legal_form",
    "publisher",
    "developer",
    "brand",
    "business_division",
    "location_of_formation",
    "creator",
    ]
    
    extracted_answers = []

    exact_extractions = 0

    for _, row in df.iterrows():

        rels_present = relationship_present(row['prediction'], relations)

        if len(rels_present) == 1:
            extracted_answers.append(rels_present[0])
            exact_extractions += 1
        else:
            answer_extraction = open_ai_request(client, generate_message_finred(row['instruction'], row['prediction'],relations))

            rels_present = relationship_present(answer_extraction, relations)

            if len(rels_present) == 1:
                extracted_answers.append(rels_present[0])
            else:
                extracted_answers.append('unknown')

    return extracted_answers


