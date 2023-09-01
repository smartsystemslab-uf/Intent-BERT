import spacy

nlp = spacy.load("en_core_web_md")  # make sure to use larger package!
doc1 = nlp("Put down screwdriver")
doc2 = nlp("Take screwdriver")

# Similarity of two documents
print(doc1, "<->", doc2, doc1.similarity(doc2))
# Similarity of tokens and spans
french_fries = doc1[0:2]
burgers = doc2[0]
print(french_fries, "<->", burgers, french_fries.similarity(burgers))