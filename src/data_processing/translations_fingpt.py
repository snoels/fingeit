finred_re_translations = {
    'Given phrases that describe the relationship between two words/phrases as options, extract the word/phrase pair and the corresponding lexical relationship between them from the input text. The output format should be "relation1: word1, word2; relation2: word3, word4". Options: product/material produced, manufacturer, distributed by, industry, position held, original broadcaster, owned by, founded by, distribution format, headquarters location, stock exchange, currency, parent organization, chief executive officer, director/manager, owner of, operator, member of, employer, chairperson, platform, subsidiary, legal form, publisher, developer, brand, business division, location of formation, creator.': 'Gegeven zinnen die de relatie beschrijven tussen twee woorden/zinnen als opties, extraheer het woord/zinpaar en de bijbehorende lexale relatie tussen hen uit de invoertekst. Het uitvoerformaat moet zijn "relatie1: woord1, woord2; relatie2: woord3, woord4". Opties zijn onder andere: product/materiaal geproduceerd, fabrikant, verdeeld door, industrie, beklede positie, originele omroep, eigendom van, opgericht door, distributieformaat, hoofdkantoor locatie, beurs, valuta, moederorganisatie, CEO, directeur/manager, eigenaar van, exploitant, lid van, werkgever, voorzitter, platform, dochteronderneming, rechtsvorm, uitgever, ontwikkelaar, merk, bedrijfsdivisie, plaats van oprichting, maker.',
    'Given the input sentence, please extract the subject and object containing a certain relation in the sentence according to the following relation types, in the format of "relation1: word1, word2; relation2: word3, word4". Relations include: product/material produced; manufacturer; distributed by; industry; position held; original broadcaster; owned by; founded by; distribution format; headquarters location; stock exchange; currency; parent organization; chief executive officer; director/manager; owner of; operator; member of; employer; chairperson; platform; subsidiary; legal form; publisher; developer; brand; business division; location of formation; creator.': 'Gegeven de invoerzin, extraheer alstublieft het onderwerp en het object met een bepaalde relatie in de zin volgens de volgende relatie types, in het formaat van "relatie1: woord1, woord2; relatie2: woord3, woord4". Relaties omvatten: product/materiaal geproduceerd; fabrikant; verdeeld door; industrie; beklede positie; originele omroep; eigendom van; opgericht door; distributieformaat; hoofdkantoor locatie; beurs; valuta; moederorganisatie; CEO; directeur/manager; eigenaar van; exploitant; lid van; werkgever; voorzitter; platform; dochteronderneming; rechtsvorm; uitgever; ontwikkelaar; merk; bedrijfsdivisie; plaats van oprichting; maker.',
}
ner_translations = {
    "Please extract entities and their types from the input sentence, entity types should be chosen from {person/organization/location}.": "Extraheer a.u.b. entiteiten en hun types uit de invoerzin, entiteitstypen moeten worden gekozen uit {persoon/organisatie/locatie}.",
}
ner_cls_translations = {
    "What is the entity type of": "Wat is het entiteitstype van",
    "With the input text as context, identify the entity type of": "Identificeer met de invoertekst als context het entiteitstype van",
    "Using the input sentence as a reference, analyze and specify the entity type of": "Analyseer met de invoerzin als referentie en specificeer het entiteitstype van",
    "In the context of the input sentence, examine and categorize the entity type of": "Onderzoek in de context van de invoerzin en categoriseer het entiteitstype van",
    "Utilize the input text as context to explore and ascertain the entity type of": "Gebruik de invoertekst als context om het entiteitstype van te verkennen en vast te stellen",
    "Leverage the input sentence to evaluate and define the entity type for": "Gebruik de invoerzin om het entiteitstype voor te evalueren en te definiëren",
    "Considering the input sentence as context, inspect and classify the entity type of": "Beschouw de invoerzin als context, inspecteer en classificeer het entiteitstype van",
    "With the input sentence as a backdrop, scrutinize and determine the entity type of": "Met de invoerzin als achtergrond, bestudeer en bepaal het entiteitstype van",
    "Interpreting the input sentence as context, specify the entity type for": "Interpreteer de invoerzin als context, specificeer het entiteitstype voor",
    "Assessing the input sentence as context, label the entity type of": "Beoordeel de invoerzin als context, label het entiteitstype van",
    "In the input sentence, determine the entity type for": "Bepaal in de invoerzin het entiteitstype voor",
    "Within the input text, identify the entity type of": "Identificeer binnen de invoertekst het entiteitstype van",
    "Analyze the input sentence to find the entity type of": "Analyseer de invoerzin om het entiteitstype van te vinden",
    "Check the input sentence for the entity type associated with": "Controleer de invoerzin voor het entiteitstype geassocieerd met",
    "Explore the input sentence to ascertain the entity type of": "Verken de invoerzin om het entiteitstype van vast te stellen",
    "Examine the input text to classify the entity type of": "Onderzoek de invoertekst om het entiteitstype van te classificeren",
    "Scrutinize the input sentence to define the entity type of": "Bestudeer de invoerzin om het entiteitstype van te definiëren",
    "in the input sentence": "in de invoerzin",
    "Options:": "Opties:",
    "person": "persoon",
    "organization": "organisatie",
    "location": "locatie",
}
finred_general = {
    "product_or_material_produced": "product/materiaal geproduceerd",
    "product or material produced": "product/materiaal geproduceerd",
    "product/material produced": "product/materiaal geproduceerd",
    "manufacturer": "fabrikant",
    "distributed_by": "verdeeld door",
    "distributed by": "verdeeld door",
    "industry": "industrie",
    "position_held": "positie bekleed",
    "position held": "positie bekleed",
    "original_broadcaster": "originele omroep",
    "original broadcaster": "originele omroep",
    "owned_by": "bezeten door",
    "owned by": "bezeten door",
    "founded_by": "opgericht door",
    "founded by": "opgericht door",
    "distribution_format": "distributieformaat",
    "distribution format": "distributieformaat",
    "headquarters_location": "hoofdkantoorlocatie",
    "headquarters location": "hoofdkantoorlocatie",
    "stock_exchange": "effectenbeurs",
    "stock exchange": "effectenbeurs",
    "currency": "valuta",
    "parent_organization": "moederorganisatie",
    "parent organization": "moederorganisatie",
    "chief_executive_officer": "chief executive officer",
    "chief executive officer": "chief executive officer",
    "director_/_manager": "directeur/manager",
    "director/manager": "directeur/manager",
    "owner_of": "eigenaar van",
    "owner of": "eigenaar van",
    "operator": "operator",
    "member_of": "lid van",
    "member of": "lid van",
    "employer": "werkgever",
    "chairperson": "voorzitter",
    "platform": "platform",
    "subsidiary": "dochteronderneming",
    "legal_form": "rechtsvorm",
    "legal form": "rechtsvorm",
    "publisher": "uitgever",
    "developer": "ontwikkelaar",
    "brand": "merk",
    "business_division": "bedrijfsdivisie",
    "business division": "bedrijfsdivisie",
    "location_of_formation": "locatie van ontstaan",
    "location of formation": "locatie van ontstaan",
    "creator": "maker",
}
finred_translations = {
    'Given phrases that describe the relationship between two words/phrases as options, extract the word/phrase pair and the corresponding lexical relationship between them from the input text. The output format should be "relation1: word1, word2; relation2: word3, word4"': 'Gegeven zinnen die de relatie tussen twee woorden/zinnen beschrijven als opties, haal het woord/zinspaar en de overeenkomstige lexicale relatie tussen hen uit de invoertekst. het formaat van het resultaat moet zijn "relatie1: woord1, woord2; relatie2: woord3, woord4"',
    'Given the input sentence, please extract the subject and object containing a certain relation in the sentence according to the following relation types, in the format of "relation1: word1, word2; relation2: word3, word4". Relations include': 'Gegeven de invoerzin, gelieve het onderwerp en het object te extraheren dat een bepaalde relatie in de zin bevat volgens de volgende soorten relaties, in het formaat van "relatie1: woord1, woord2; relatie2: woord3, woord4". Relaties zijn onder andere',
    "Utilize the input text as a context reference, choose the right relationship between": "Gebruik de invoertekst als een contextreferentie, kies de juiste relatie tussen",
    " and ": " en ",
    "Options:": "Opties:",
    "from the options": "uit de opties",
    "What is the relationship between": "Wat is de relatie tussen",
    "in the context of the input sentence. Choose an answer from": "in de context van de invoerzin. Kies een antwoord uit",
}
finred_cls_translations = {
    "Utilize the input text as a context reference, choose the right relationship between": "Gebruik de invoertekst als contextreferentie, kies de juiste relatie tussen",
    "from the options": "uit de opties",
    "Refer to the input text as context and select the correct relationship between": "Verwijs naar de invoertekst als context en selecteer de juiste relatie tussen",
    "from the available options": "uit de beschikbare opties",
    "Take context from the input text and decide on the accurate relationship between": "Haal de context uit de invoertekst en beslis over de nauwkeurige relatie tussen",
    "from the options provided": "uit de verstrekte opties",
    "What is the relationship between": "Wat is de relatie tussen",
    "in the context of the input sentence": "in de context van de invoerzin",
    "In the context of the input sentence, determine the relationship between": "In de context van de invoerzin, bepaal de relatie tussen",
    "Analyze the relationship between": "Analyseer de relatie tussen",
    "within the context of the input sentence": "binnen de context van de invoerzin",
    " and ": " en ",
    "Options:": "Opties:",
}

headline_translations = {
    "Does the news headline talk about price": "Gaat de krantenkop over de prijs",
    "Does the news headline talk about price going up": "Gaat de krantenkop over de prijs die omhoog gaat",
    "Does the news headline talk about price staying constant": "Gaat de krantenkop over de prijs die constant blijft",
    "Does the news headline talk about price going down": "Gaat de krantenkop over de prijs die omlaag gaat",
    "Does the news headline talk about price in the past": "Gaat de krantenkop over de prijs in het verleden",
    "Does the news headline talk about price in the future": "Gaat de krantenkop over de prijs in de toekomst",
    "Does the news headline talk about a general event (apart from prices) in the past": "Gaat de krantenkop over een algemene gebeurtenis (afgezien van de prijzen) in het verleden",
    "Does the news headline talk about a general event (apart from prices) in the future": "Gaat de krantenkop over een algemene gebeurtenis (afgezien van de prijzen) in de toekomst",
    "Does the news headline compare gold with any other asset": "Vergelijkt de krantenkop goud met een ander actief",
    "Please choose an answer from {Yes/No}": "Kies alstublieft een antwoord uit {Ja/Nee}",
}

sentiment_translations = {
    "What is the sentiment of this tweet? Please choose an answer from {negative/neutral/positive}.": "Wat is het sentiment van deze tweet? Kies een antwoord uit {negatief/neutraal/positief}.",
    "What is the sentiment of this news? Please choose an answer from {negative/neutral/positive}.": "Wat is het sentiment van dit nieuws? Kies een antwoord uit {negatief/neutraal/positief}.",
    "What is the sentiment of this news? Please choose an answer from {strong negative/moderately negative/mildly negative/neutral/mildly positive/moderately positive/strong positive}.": "Wat is het sentiment van dit nieuws? Kies een antwoord uit {sterk negatief/matig negatief/mild negatief/neutraal/mild positief/matig positief/sterk positief}.",
}
