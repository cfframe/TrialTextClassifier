# https://course.spacy.io/en/chapter1
# https://github.com/ines/spacy-course
import json
import spacy
from spacy.language import Language
from spacy.matcher import Matcher
from spacy.matcher import PhraseMatcher
from spacy.tokens import Doc, Span, Token


def print_header(text: str):
    print(text)
    print('-' * len(text))


def ch01_02():
    print_header('1.2 Getting Started')

    # Create the English nlp object
    nlp = spacy.blank('en')

    # Process a text
    doc = nlp('This is a sentence.')

    # Print the document text
    print(doc.text)

    # Create the German nlp object
    nlp = spacy.blank('de')

    # Process a text (this is German for: "Kind regards!")
    doc = nlp('Liebe Grüße!')

    # Print the document text
    print(doc.text)

    # Create the Spanish nlp object
    nlp = spacy.blank('es')

    # Process a text (this is Spanish for: "How are you?")
    doc = nlp("¿Cómo estás?")

    # Print the document text
    print(doc.text)


def ch01_03():
    print_header('1.3 Documents, spans and tokens')

    nlp = spacy.blank('en')

    # Process the text
    doc = nlp('I like tree kangaroos and narwhals.')

    # Select the first token
    first_token = doc[0]

    # Print the first token's text
    print(first_token.text)

    # A slice of the Doc for "tree kangaroos"
    tree_kangaroos = doc[2:4]
    print(tree_kangaroos.text)

    # A slice of the Doc for "tree kangaroos and narwhals" (without the ".")
    tree_kangaroos_and_narwhals = doc[2:-1]
    print(tree_kangaroos_and_narwhals.text)


def ch01_04():
    print_header('1.4 Lexical attributes')

    nlp = spacy.blank('en')

    # Process the text
    doc = nlp(
        'In 1990, more than 60% of people in East Asia were in extreme poverty. '
        'Now less than 4% are.'
    )

    # Iterate over the tokens in the doc
    for token in doc:
        # Check if the token resembles a number
        if token.like_num:
            # Get the next token in the document
            next_token = doc[token.i + 1]
            # Check if the next token's text equals "%"
            if next_token.text == "%":
                print("Percentage found:", token.text)


def ch01_07():
    print_header('1.7 Loading pipelines')

    # Load the 'en_core_web_sm' pipeline
    nlp = spacy.load('en_core_web_sm')

    text = 'It’s official: Apple is the first U.S. public company to reach a $1 trillion market value'

    # Process the text
    doc = nlp(text)

    # Print the document text
    print(doc.text)


def ch01_08():
    print_header('1.8 Predicting linguistic annotations')

    nlp = spacy.load('en_core_web_sm')

    text = 'It’s official: Apple is the first U.S. public company to reach a $1 trillion market value'

    # Process the text
    doc = nlp(text)

    print('\nIterate over the tokens - token text, part-of-speech tag and dependency label\n')
    for token in doc:
        # Get the token text, part-of-speech tag and dependency label
        token_text = token.text
        token_pos = token.pos_  # pos gives numeric value, pos_ gives textual
        exp_pos = spacy.explain(token_pos)
        token_dep = token.dep_  # dep gives numeric value, dep_ gives textual
        exp_dep = spacy.explain(token_dep)

        # This is for formatting only
        print(f"{token_text:<12}{token_pos:<10}{exp_pos:<15}{token_dep:<10}{exp_dep}")

    print('\nIterate over the predicted entities\n')

    for ent in doc.ents:
        # Print the entity text and its label
        print(ent.text, ent.label)


def ch01_09():
    print_header('1.9 Predicting named entities in context')

    nlp = spacy.load('en_core_web_sm')

    text = 'Upcoming iPhone X release date leaked as Apple reveals pre-orders'

    # Process the text
    doc = nlp(text)

    # Iterate over the entities
    for ent in doc.ents:
        # Print the entity text and label
        print(ent.text, ent.label)

    # Get the span for 'iPhone X'
    iphone_x = doc[1:3]

    # Print the span text
    print('Missing entity:', iphone_x.text)


def ch01_11():

    print_header('1.11 Rule based matching')
    nlp = spacy.load('en_core_web_sm')
    text = 'Upcoming iPhone X release date leaked as Apple reveals pre-orders'
    doc = nlp(text)

    # Initialize the Matcher with the shared vocabulary
    matcher = Matcher(nlp.vocab)

    # Create a pattern matching TEXT values of two consecutive tokens: "iPhone" and "X"
    # Note: Dictionary entry type (e.g. 'TEXT') is not case sensitive, but the values are (e.g. 'iPhone').
    pattern = [{'TEXT': 'iPhone'}, {'TEXT': 'X'}]

    # Add the pattern to the matcher (as a list of patterns, but only one pattern here)
    matcher.add("IPHONE_X_PATTERN", [pattern])

    # Use the matcher on the doc
    matches = matcher(doc)

    print("Matches:", [doc[start:end].text for match_id, start, end in matches])


def ch01_12():
    print_header('1.12 Writing match patterns')

    print('Part 1')
    print('Write one pattern that only matches mentions of the full iOS versions: “iOS 7”, “iOS 11” and “iOS 10”.')

    nlp = spacy.load('en_core_web_sm')
    matcher = Matcher(nlp.vocab)

    doc = nlp(
        "After making the iOS update you won't notice a radical system-wide "
        "redesign: nothing like the aesthetic upheaval we got with iOS 7. Most of "
        "iOS 11's furniture remains the same as in iOS 10. But you will discover "
        "some tweaks once you delve a little deeper."
    )

    # Write a pattern for full iOS versions ("iOS 7", "iOS 11", "iOS 10")
    pattern = [{"TEXT": "iOS"}, {"IS_DIGIT": True}]

    # Add the pattern to the matcher and apply the matcher to the doc
    matcher.add("IOS_VERSION_PATTERN", [pattern])
    matches = matcher(doc)
    print("Total matches found:", len(matches))

    # Iterate over the matches and print the span text
    for match_id, start, end in matches:
        print("Match found:", doc[start:end].text)

    print('\nPart 2\n')
    print('Write one pattern that only matches forms of “download” (tokens with the lemma “download”), followed by a '
          'token with the part-of-speech tag "PROPN" (proper noun).')

    nlp = spacy.load('en_core_web_sm')
    matcher = Matcher(nlp.vocab)

    doc = nlp(
        "i downloaded Fortnite on my laptop and can't open the game at all. Help? "
        "so when I was downloading Minecraft, I got the Windows version where it "
        "is the '.zip' folder and I used the default program to unpack it... do "
        "I also need to download Winzip?"
    )

    # Write a pattern that matches a form of "download" plus proper noun
    # Values for LEMMA and POS entries are case-sensitive
    pattern = [{"LEMMA": "download"}, {"POS": "PROPN"}]

    # Add the pattern to the matcher and apply the matcher to the doc
    matcher.add("DOWNLOAD_THINGS_PATTERN", [pattern])
    matches = matcher(doc)
    print("Total matches found:", len(matches))

    # Iterate over the matches and print the span text
    for match_id, start, end in matches:
        print("Match found:", doc[start:end].text)

    print('\nPart 3\n')
    print('Write one pattern that matches adjectives ("ADJ") followed by one or two "NOUN"s (one noun and one optional noun).')

    matcher = Matcher(nlp.vocab)

    doc = nlp(
        "Features of the app include a beautiful design, smart search, automatic "
        "labels and optional voice responses."
    )

    # Write a pattern for adjective plus one or two nouns
    pattern = [{"POS": "ADJ"}, {"POS": "NOUN"}, {"POS": "NOUN", "OP": "?"}]

    # Add the pattern to the matcher and apply the matcher to the doc
    matcher.add("ADJ_NOUN_PATTERN", [pattern])
    matches = matcher(doc)
    print("Total matches found:", len(matches))

    # Iterate over the matches and print the span text
    for match_id, start, end in matches:
        print("Match found:", doc[start:end].text)


def ch02_02():
    print_header('2.2 Strings to hashes')

    print('\nPart 1\n')
    print('Look up the string “cat” in nlp.vocab.strings to get the hash.')

    nlp = spacy.blank("en")
    # 'doc = nlp("I have a cat")' initialises the nlp for use for all the words of interest.
    # Alternatively, could just call 'nlp("I have a cat")' assigning to the doc variable.

    doc = nlp("I have a cat")

    # Look up the hash for the word "cat"
    cat_hash = nlp.vocab.strings['cat']

    print(cat_hash)

    # Look up the cat_hash to get the string
    print('Look up the hash to get back the string.')
    cat_string = nlp.vocab.strings[cat_hash]
    print(cat_string)

    print('\nPart 2\n')
    print('Look up the string label “PERSON” in nlp.vocab.strings to get the hash.')
    nlp = spacy.blank("en")
    doc = nlp("David Bowie is a PERSON")

    # Look up the hash for the string label "PERSON"
    person_hash = nlp.vocab.strings["PERSON"]
    print(person_hash)

    # Look up the person_hash to get the string
    person_string = nlp.vocab.strings[person_hash]
    print(person_string)


def ch02_03():
    print_header('2.3 Vocab, hashes and lexemes')

    # Create an English and German nlp object
    nlp = spacy.blank("en")
    nlp_de = spacy.blank("de")

    # Get the ID for the string 'Bowie'
    bowie_id = nlp.vocab.strings["Bowie"]
    print(bowie_id)

    # Look up the ID for "Bowie" in the vocab
    # THIS WOULD FAIL because bowie_id has not been initialised in nlp_de
    # print(nlp_de.vocab.strings[bowie_id])

    bowie_id_de = nlp_de.vocab.strings["Bowie"]
    print(bowie_id_de)


def ch02_05():
    print_header('2.5 Creating a Doc')

    nlp = spacy.blank("en")

    print('\nPart 1\n')

    # Desired text: "spaCy is cool!"
    words = ["spaCy", "is", "cool", "!"]
    # spaces indicates whether word is followed by a space
    spaces = [True, True, False, False]

    # Create a Doc from the words and spaces
    doc = Doc(nlp.vocab, words=words, spaces=spaces)
    print(doc.text)

    print('\nPart 2\n')

    # Desired text: "Go, get started!"
    words = ["Go", ",", "get", "started", "!"]
    spaces = [False, True, True, False, False]

    # Create a Doc from the words and spaces
    doc = Doc(nlp.vocab, words=words, spaces=spaces)
    print(doc.text)

    print('\nPart 3\n')
    # Desired text: "Oh, really?!"
    words = ['Oh', ',', 'really', '?', '!']
    spaces = [False, True, False, False, False]

    # Create a Doc from the words and spaces
    doc = Doc(nlp.vocab, words=words, spaces=spaces)
    print(doc.text)


def ch02_06():
    print_header('2.6 Docs, spans and entities from scratch')

    nlp = spacy.blank("en")

    print('\nPart 1\n')

    words = ["I", "like", "David", "Bowie"]
    spaces = [True, True, True, False]

    # Create a doc from the words and spaces
    doc = Doc(nlp.vocab, words=words, spaces=spaces)
    print(doc.text)

    # Create a span for "David Bowie" from the doc and assign it the label "PERSON"
    span = Span(doc=doc, start=2, end=4, label='PERSON')
    print(span.text, span.label_)

    # Add the span to the doc's entities
    doc.ents = [span]

    # Print entities' text and labels
    print([(ent.text, ent.label_) for ent in doc.ents])


def ch02_07():
    print_header('2.7 Data structures best practices')

    nlp = spacy.load("en_core_web_sm")
    doc = nlp("Berlin looks like a nice city")

    for token in doc:
        # Check if the current token is a proper noun
        if token.pos_ == "PROPN":
            # Check if the next token is a verb
            if token.i + 1 < len(doc) and doc[token.i + 1].pos_ == "VERB":
                print("Found proper noun before a verb:", token.text)


def ch02_09():
    print_header('2.9 Inspecting word vectors')

    # Load the medium "en_core_web_md" pipeline with word vectors
    nlp = spacy.load("en_core_web_md")

    # Process a text
    doc = nlp("Two bananas in pyjamas")

    # Get the vector for the token "bananas"
    bananas_vector = doc[1].vector
    print(bananas_vector)


def ch02_10():
    print_header('2.10 Comparing similarities')

    # Load the medium "en_core_web_md" pipeline with word vectors
    nlp = spacy.load("en_core_web_md")

    doc1 = nlp("It's a warm summer day")
    doc2 = nlp("It's sunny outside")

    print('\nPart 1\n')

    # Get the similarity of doc1 and doc2
    # 1.0 means identical?
    # Order of docs does not matter i.e. doc1.similarity(doc2) and doc2.similarity(doc1) give same result.
    similarity = doc1.similarity(doc2)
    print(similarity)

    print('\nPart 2\n')

    doc = nlp("TV and books")
    token1, token2 = doc[0], doc[2]

    # Get the similarity of the tokens "TV" and "books"
    similarity = token1.similarity(token2)
    print(similarity)

    print('\nPart 3\n')

    doc = nlp("This was a great restaurant. Afterwards, we went to a really nice bar.")

    # Create spans for "great restaurant" and "really nice bar"
    span1 = doc[3:5]
    span2 = doc[12:-1]

    # Get the similarity of the spans
    similarity = span1.similarity(span2)
    print(similarity)


def ch02_13():
    print_header('2.13 Debugging patterns(2)')

    nlp = spacy.load("en_core_web_sm")
    doc = nlp(
        "Twitch Prime, the perks program for Amazon Prime members offering free "
        "loot, games and other benefits, is ditching one of its best features: "
        "ad-free viewing. According to an email sent out to Amazon Prime members "
        "today, ad-free viewing will no longer be included as a part of Twitch "
        "Prime for new members, beginning on September 14. However, members with "
        "existing annual subscriptions will be able to continue to enjoy ad-free "
        "viewing until their subscription comes up for renewal. Those with "
        "monthly subscriptions will have access to ad-free viewing until October 15."
    )

    # Create the match patterns
    pattern1 = [{"LOWER": "amazon"}, {"IS_TITLE": True, "POS": "PROPN"}]
    pattern2 = [{"LOWER": "ad"}, {"TEXT": "-"}, {"LOWER": "free"}, {"POS": "NOUN"}]

    # Initialize the Matcher and add the patterns
    matcher = Matcher(nlp.vocab)
    matcher.add("PATTERN1", [pattern1])
    matcher.add("PATTERN2", [pattern2])

    # Iterate over the matches
    for match_id, start, end in matcher(doc):
        # Print pattern string name and text of matched span
        print(doc.vocab.strings[match_id], doc[start:end].text)


def ch02_14():
    print_header('2.14 Efficient phrase matching')

    with open("exercises/en/countries.json", encoding="utf8") as f:
        COUNTRIES = json.loads(f.read())

    nlp = spacy.blank("en")
    doc = nlp("Czech Republic may help Slovakia protect its airspace")

    # Initialize the PhraseMatcher
    matcher = PhraseMatcher(nlp.vocab)

    # Create pattern Doc objects and add them to the matcher
    # This is the faster version of: [nlp(country) for country in COUNTRIES]
    patterns = list(nlp.pipe(COUNTRIES))
    matcher.add("COUNTRY", patterns)

    # Call the matcher on the test document and print the result
    matches = matcher(doc)
    print([doc[start:end] for match_id, start, end in matches])


def ch02_15():
    print_header('2.15 Extracting countries and relationships')

    with open("exercises/en/countries.json", encoding="utf8") as f:
        COUNTRIES = json.loads(f.read())
    with open("exercises/en/country_text.txt", encoding="utf8") as f:
        TEXT = f.read()

    nlp = spacy.load("en_core_web_sm")
    matcher = PhraseMatcher(nlp.vocab)
    patterns = list(nlp.pipe(COUNTRIES))
    matcher.add("COUNTRY", patterns)

    # Create a doc and reset existing entities
    doc = nlp(TEXT)
    doc.ents = []

    # Iterate over the matches
    for match_id, start, end in matcher(doc):
        # Create a Span with the label for "GPE" (geopolitical entity)
        span = Span(doc, start, end, label="GPE")

        # Overwrite the doc.ents and add the span
        doc.ents = list(doc.ents) + [span]

        # Get the span's root head token. This is the 'introductory' word of the phrase (or partial phrase)
        # relating to the span, though can come after e.g. 'in Namibia', 'invaded Iraq', 'Haiti earthquake'
        # Cf left_edge and right_edge.
        span_root_head = span.root.head
        # Print the text of the span root's head token and the span text
        print(span_root_head.text, "-->", span.text)

    # Print the entities in the document
    print([(ent.text, ent.label_) for ent in doc.ents if ent.label_ == "GPE"])


def ch03_03():
    print_header('3.3 Inspecting the pipeline')

    # Load the en_core_web_sm pipeline
    nlp = spacy.load('en_core_web_sm')

    # Print the names of the pipeline components
    print(nlp.pipe_names)

    # Print the full pipeline of (name, component) tuples
    print(nlp.pipeline)


def ch03_06():
    print_header('3.6 Simple components')

    # Define the custom component
    @Language.component("length_component")
    def length_component_function(doc):
        # Get the doc's length
        doc_length = len(doc)
        print(f"This document is {doc_length} tokens long.")
        # Return the doc
        return doc


    # Load the small English pipeline
    nlp = spacy.load("en_core_web_sm")

    # Add the component first in the pipeline and print the pipe names
    nlp.add_pipe('length_component', first=True)
    print(nlp.pipe_names)

    # Process a text
    doc = nlp('This is a sentence')


def ch03_07():
    print_header('3.7 Complex components')

    nlp = spacy.load("en_core_web_sm")
    animals = ["Golden Retriever", "cat", "turtle", "Rattus norvegicus"]
    animal_patterns = list(nlp.pipe(animals))
    print("animal_patterns:", animal_patterns)
    matcher = PhraseMatcher(nlp.vocab)
    matcher.add("ANIMAL", animal_patterns)

    # Define the custom component
    @Language.component("animal_component")
    def animal_component_function(doc):
        # Apply the matcher to the doc
        matches = matcher(doc)
        # Create a Span for each match and assign the label "ANIMAL"
        spans = [Span(doc, start, end, label="ANIMAL") for match_id, start, end in matches]
        # Overwrite the doc.ents with the matched spans
        doc.ents = spans
        return doc

    # Add the component to the pipeline after the "ner" component
    nlp.add_pipe('animal_component', after='ner')
    print(nlp.pipe_names)

    # Process the text and print the text and label for the doc.ents
    doc = nlp("I have a cat and a Golden Retriever")
    print([(ent.text, ent.label_) for ent in doc.ents])


def ch03_09():
    print_header('3.9 Setting extension attributes (1)')

    print('\nStep 1\n')

    nlp = spacy.blank("en")

    # Register the Token extension attribute "is_country" with the default value False
    Token.set_extension('is_country', default=False)

    # Process the text and set the is_country attribute to True for the token "Spain"
    doc = nlp("I live in Spain.")
    doc[3]._.is_country = True

    # Print the token text and the is_country attribute for all tokens
    print([(token.text, token._.is_country) for token in doc])

    print('\nStep 2\n')

    nlp = spacy.blank("en")

    # Define the getter function that takes a token and returns its reversed text
    def get_reversed(token):
        return token.text[::-1]


    # Register the Token property extension "reversed" with the getter get_reversed
    Token.set_extension('reversed', getter=get_reversed)

    # Process the text and print the reversed attribute for each token
    doc = nlp("All generalizations are false, including this one.")
    for token in doc:
        print("reversed:", token._.reversed)


def ch03_10():
    print_header('3.10 Setting extension attributes (2)')

    print('\nStep 1\n')

    nlp = spacy.blank("en")

    # Define the getter function
    def get_has_number(doc):
        # Return if any of the tokens in the doc return True for token.like_num
        return any(token.like_num for token in doc)

    # Register the Doc property extension "has_number" with the getter get_has_number
    Doc.set_extension('has_number', getter=get_has_number)

    # Process the text and check the custom has_number attribute
    doc = nlp("The museum closed for five years in 2012.")
    print("has_number:", doc._.has_number)

    print('\nStep 2\n')

    nlp = spacy.blank("en")

    # Define the method
    def to_html(span, tag):
        # Wrap the span text in a HTML tag and return it
        return f"<{tag}>{span.text}</{tag}>"

    # Register the Span method extension "to_html" with the method to_html
    Span.set_extension('to_html', method=to_html)

    # Process the text and call the to_html method on the span with the tag name "strong"
    doc = nlp("Hello world, this is a sentence.")
    span = doc[0:2]
    print(span._.to_html('strong'))


def ch03_11():
    print_header('3.11 Entities and extensions')

    nlp = spacy.load("en_core_web_sm")

    def get_wikipedia_url(span):
        # Get a Wikipedia URL if the span has one of the labels
        if span.label_ in ("PERSON", "ORG", "GPE", "LOCATION"):
            entity_text = span.text.replace(" ", "_")
            return "https://en.wikipedia.org/w/index.php?search=" + entity_text

    # Set the Span extension wikipedia_url using the getter get_wikipedia_url
    Span.set_extension('wikipedia_url', getter=get_wikipedia_url)

    doc = nlp(
        "In over fifty years from his very first recordings right through to his "
        "last album, David Bowie was at the vanguard of contemporary culture."
    )
    for ent in doc.ents:
        # Print the text and Wikipedia URL of the entity
        print(ent.text, ent._.wikipedia_url)


def ch03_12():
    print_header('3.12 Components with extensions')

    with open("exercises/en/countries.json", encoding="utf8") as f:
        COUNTRIES = json.loads(f.read())

    with open("exercises/en/capitals.json", encoding="utf8") as f:
        CAPITALS = json.loads(f.read())

    nlp = spacy.blank("en")
    matcher = PhraseMatcher(nlp.vocab)
    matcher.add("COUNTRY", list(nlp.pipe(COUNTRIES)))

    @Language.component("countries_component")
    def countries_component_function(doc):
        # Create an entity Span with the label "GPE" for all matches
        matches = matcher(doc)
        doc.ents = [Span(doc, start, end, label='GPE') for match_id, start, end in matches]
        return doc

    # Add the component to the pipeline
    nlp.add_pipe("countries_component")
    print(nlp.pipe_names)

    # Getter that looks up the span text in the dictionary of country capitals
    get_capital = lambda span: CAPITALS.get(span.text)

    # Register the Span extension attribute "capital" with the getter get_capital
    Span.set_extension("capital", getter=get_capital)

    # Process the text and print the entity text, label and capital attributes
    doc = nlp("Czech Republic may help Slovakia protect its airspace")
    print([(ent.text, ent.label_, ent._.capital) for ent in doc.ents])


def ch03_14():
    print_header('3.14 Processing streams')

    print('\nPart 1\n')

    nlp = spacy.load("en_core_web_sm")

    with open("exercises/en/tweets.json", encoding="utf8") as f:
        TEXTS = json.loads(f.read())

    # Process the texts and print the adjectives
    # FROM
    # for text in TEXTS:
    #     doc = nlp(text)
    for doc in nlp.pipe(TEXTS):
        print([token.text for token in doc if token.pos_ == "ADJ"])

    print('\nPart 2\n')

    nlp = spacy.load("en_core_web_sm")

    with open("exercises/en/tweets.json", encoding="utf8") as f:
        TEXTS = json.loads(f.read())

    # Process the texts and print the entities
    # FROM
    # docs = [nlp(text) for text in TEXTS]
    docs = list(nlp.pipe(TEXTS))
    entities = [doc.ents for doc in docs]
    print(*entities)

    print('\nPart 3\n')

    nlp = spacy.blank("en")

    people = ["David Bowie", "Angela Merkel", "Lady Gaga"]

    # Create a list of patterns for the PhraseMatcher
    # FROM
    # patterns = [nlp(person) for person in people]
    patterns = list(nlp.pipe(people))


def ch03_15():
    print_header('3.15 Processing data with context')

    with open("exercises/en/bookquotes.json", encoding="utf8") as f:
        DATA = json.loads(f.read())

    nlp = spacy.blank("en")

    # Register the Doc extension "author" (default None)
    Doc.set_extension("author", default=None)

    # Register the Doc extension "book" (default None)
    Doc.set_extension("book", default=None)

    for doc, context in nlp.pipe(DATA, as_tuples=True):
        # Set the doc._.book and doc._.author attributes from the context
        doc._.book = context["book"]
        doc._.author = context["author"]

        # Print the text and custom attribute data
        print(f"{doc.text}\n — '{doc._.book}' by {doc._.author}\n")


def ch03_16():
    print_header('3.16 Selective processing')

    print('\nPart 1\n')

    nlp = spacy.load("en_core_web_sm")
    text = (
        "Chick-fil-A is an American fast food restaurant chain headquartered in "
        "the city of College Park, Georgia, specializing in chicken sandwiches."
    )

    # Only tokenize the text
    # FROM
    # doc = nlp(text)
    doc = nlp.make_doc(text)
    print([token.text for token in doc])

    print('\nPart 2\n')

    nlp = spacy.load("en_core_web_sm")
    text = (
        "Chick-fil-A is an American fast food restaurant chain headquartered in "
        "the city of College Park, Georgia, specializing in chicken sandwiches."
    )

    # Disable the tagger and lemmatizer
    with nlp.select_pipes(disable=['tagger', 'lemmatizer']):
        # Process the text
        doc = nlp(text)
        # Print the entities in the doc
        print(doc.ents)


def test_read_file():
    with open("exercises/en/tweets.json", encoding="utf8", mode='r') as f:
        COUNTRIES = f.read()
        print(COUNTRIES)


if __name__ == '__main__':
    #test_read_file()
    ch03_16()
