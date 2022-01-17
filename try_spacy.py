# https://course.spacy.io/en/chapter1
import spacy
from spacy.matcher import Matcher


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
    # Note: TEXT in each dictionary entry is NOT case sensitive, but the values are (e.g. 'iPhone').
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
    # Both LEMMA and POS are case-sensitive
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


if __name__ == '__main__':
    ch01_12()
