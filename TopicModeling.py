import os
import pymupdf
import re

import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk import pos_tag, word_tokenize

from gensim.models import Phrases
from gensim.corpora import Dictionary
from gensim.models import LdaModel

# nltk.download('averaged_perceptron_tagger_eng')

directory = '/Users/db/Desktop/Fordham/Spring 2025/Agentic AI/VNR Reports (English)'

docs = []

#Creating dictionaries of country specific terms to remove during preprocessing
countries = [
    "afghanistan", "albania", "algeria", "andorra", "angola", "antigua and barbuda", "argentina", 
    "armenia", "australia", "austria", "azerbaijan", "bahamas", "bahrain", "bangladesh", "barbados", 
    "belarus", "belgium", "belize", "benin", "bhutan", "bolivia", "bosnia and herzegovina", "botswana", 
    "brazil", "brunei", "bulgaria", "burkina faso", "burundi", "cabo verde", "cambodia", "cameroon", 
    "canada", "central african republic", "chad", "chile", "china", "colombia", "comoros", "congo", 
    "congo (democratic republic of the)", "costa rica", "croatia", "cuba", "cyprus", "czech republic", 
    "denmark", "djibouti", "dominica", "dominican republic", "ecuador", "egypt", "el salvador", "equatorial guinea", 
    "eritrea", "estonia", "eswatini", "ethiopia", "fiji", "finland", "france", "gabon", "gambia", "georgia", 
    "germany", "ghana", "greece", "grenada", "guatemala", "guinea", "guinea-bissau", "guyana", "haiti", "honduras", 
    "hungary", "iceland", "india", "indonesia", "iran", "iraq", "ireland", "israel", "italy", "jamaica", "japan", 
    "jordan", "kazakhstan", "kenya", "kiribati", "korea (north)", "korea (south)", "kuwait", "kyrgyzstan", "laos", 
    "latvia", "lebanon", "lesotho", "liberia", "libya", "liechtenstein", "lithuania", "luxembourg", "madagascar", 
    "malawi", "malaysia", "maldives", "mali", "malta", "marshall islands", "mauritania", "mauritius", "mexico", 
    "micronesia", "moldova", "monaco", "mongolia", "montenegro", "morocco", "mozambique", "myanmar", "namibia", 
    "nauru", "nepal", "netherlands", "new zealand", "nicaragua", "niger", "nigeria", "north macedonia", "norway", 
    "oman", "pakistan", "palau", "panama", "papua new guinea", "paraguay", "peru", "philippines", "poland", 
    "portugal", "qatar", "romania", "russia", "rwanda", "saint kitts and nevis", "saint lucia", "saint vincent and the grenadines", 
    "samoa", "san marino", "sao tome and principe", "saudi arabia", "senegal", "serbia", "seychelles", "sierra leone", 
    "singapore", "slovakia", "slovenia", "solomon islands", "somalia", "south africa", "south sudan", "spain", 
    "sri lanka", "sudan", "suriname", "sweden", "switzerland", "syria", "taiwan", "tajikistan", "tanzania", 
    "thailand", "timor-leste", "togo", "tonga", "trinidad and tobago", "tunisia", "turkey", "turkmenistan", 
    "tuvalu", "uganda", "ukraine", "united arab emirates", "united kingdom", "united states", "uruguay", 
    "uzbekistan", "vanuatu", "vatican city", "venezuela", "vietnam", "yemen", "zambia", "zimbabwe", 
    "palestine", 'emirate', 'lanka'
]

regions = [
    "caribbean", "sub-saharan africa", "north africa", "middle east", "latin america", 
    "europe", "north america", "central asia", "eastern asia", "southern asia", 
    "southeast asia", "oceania", "eastern europe", "western europe", "southern europe", 
    "northern europe", "central europe", "south asia", "central america", "western asia", 
    "pacific islands", "arab world", "indian ocean", "south pacific", "indochina", 
    "central africa", "western africa", "eastern africa", "nordic countries", "baltic states", 
    "balkans", "scandinavia", "sahel", "caucasus", "great lakes region", "horn of africa", 
    "caribbean sea", "andalusia", "southern cone", "subarctic", "amazon basin", "mediterranean"
]

nationality = [
    "afghan", "albanian", "algerian", "andorran", "angolan", "antiguan and barbudan", "argentine", 
    "armenian", "australian", "austrian", "azerbaijani", "bahamian", "bahraini", "bangladeshi", "barbadian", 
    "belarusian", "belgian", "belizean", "beninese", "bhutani", "bolivian", "bosnian and herzegovinian", "botswanan", 
    "brazilian", "bruneian", "bulgarian", "burkinabe", "burundian", "cabo verdean", "cambodian", "cameroonian", 
    "canadian", "central african republican", "chadian", "chilean", "chinese", "colombian", "comorian", "congolese", 
    "congolese (democratic republic of the)", "costarican", "croatian", "cuban", "cypriot", "czech", 
    "danish", "djiboutian", "dominican", "dominican republican", "ecuadorian", "egyptian", "el salvadoran", "equatorial guinean", 
    "eritrean", "estonian", "eswatinian", "ethiopian", "fijian", "finnish", "french", "gabonese", "gambian", "georgian", 
    "german", "ghanian", "greek", "grenadian", "guatemalan", "guinean", "guinea-bissauan", "guyanese", "haitian", "honduran", 
    "hungarian", "icelandic", "indian", "indonesian", "iranian", "iraqi", "irish", "israeli", "italian", "jamaican", "japanese", 
    "jordanian", "kazakhstani", "kenyan", "kiribatian", "north korean", "south korean", "kuwaiti", "kyrgyzstani", "laotian", 
    "latvian", "lebanese", "lesothian", "liberian", "libyan", "liechtensteinian", "lithuanian", "luxembourgian", "madagascan", 
    "malawian", "malaysian", "maldivian", "malian", "maltese", "marshallese", "mauritanian", "mauritian", "mexican", 
    "micronesian", "moldovan", "monacan", "mongolian", "montenegrin", "moroccan", "mozambican", "myanmarian", "namibian", 
    "nauruan", "nepali", "dutch", "new zealander", "nicaraguan", "nigerien", "nigerian", "north macedonian", "norwegian", 
    "omanian", "pakistani", "palauan", "panamanian", "papua new guinean", "paraguayan", "peruvian", "filipino", "polish", 
    "portuguese", "qatari", "romanian", "russian", "rwandan", "saint kitts and nevisian", "saint lucian", "saint vincentian", 
    "samoan", "san marinese", "sao tomean", "saudi arabian", "senegalese", "serbian", "seychellois", "sierra leonean", 
    "singaporean", "slovak", "slovene", "solomon islander", "somalian", "south african", "south sudanese", "spanish", 
    "sri lankan", "sudanese", "surinamer", "swedish", "swiss", "syrian", "taiwanese", "tajik", "tanzanian", 
    "thai", "timorese", "togolese", "tongan", "trinidadian and tobagonian", "tunisian", "turkish", "turkmen", 
    "tuvaluan", "ugandan", "ukrainian", "emirati", "british", "american", "uruguayan", 
    "uzbek", "vanuatuan", "vatican", "venezuelan", "vietnamese", "yemeni", "zambian", "zimbabwean", 
    "palestinian"
]


currencies = [
    "afghan afghani", "lek", "algerian dinar", "euro", " kwanza", "east caribbean dollar", "argentine peso",
    "dram", "australian dollar", "austrian schilling", "azerbaijani manat", "bahamian dollar", "bahraini dinar",
    "taka", "barbadian dollar", "belarusian ruble", "euro", "belize dollar", "west african CFA franc", 
    "bhutanese ngultrum", "bolivian boliviano", "bosnia and herzegovina convertible mark", "botswana pula",
    "brazilian real", "brunei dollar", "bulgarian lev", "burkina faso CFA franc", "burundian franc", 
    "cape verde escudo", "cambodian riel", "central african CFA franc", "chilean peso", "yuan", "colombian peso", 
    "comorian franc", "congolese franc", "congolese franc", "costarican colón", "croatian kuna", "cuban peso",
    "cypriot pound", "czech koruna", "danish krone", "djiboutian franc", "east caribbean dollar", "dominican peso", 
    "united states dollar", "egyptian pound", "el salvadoran colón", "equatorial guinean ekwele", "nakfa", 
    "estonian kroon", "eswatini lilangeni", "ethiopian birr", "fijian dollar", "euro", "gabonese franc", 
    "gambian dalasi", "georgian lari", "german mark", "ghanaian cedi", "euro", "greek drachma", 
    "barbadian dollar", "guatemalan quetzal", "guinean franc", "guinea-bissau peso", "guyanese dollar", 
    "haitian gourde", "honduran lempira", "hungarian forint", "icelandic króna", "indian rupee", 
    "indonesian rupiah", "iranian rial", "iraqi dinar", "irish pound", "israeli new shekel", "italian lira",
    "jamaican dollar", "japanese yen", "jordanian dinar", "kazakhstani tenge", "kenyan shilling", 
    "kiribati dollar", "north korean won", "south korean won", "kuwaiti dinar", "kyrgyzstani som", 
    "laotian kip", "latvian lats", "lebanese pound", "lesotho loti", "liberian dollar", "libyan dinar",
    "liechtenstein franc", "lithuanian litas", "luxembourg franc", "malagasy ariary", "malawian kwacha", 
    "malaysian ringgit", "maldivian rufiyaa", "malian franc", "euro", "marshall islands dollar", "mauritian rupee", 
    "mexican peso", "micronesian dollar", "moldovan leu", "monégasque franc", "mongolian tugrik", 
    "montenegrin dinar", "moroccan dirham", "mozambican metical", "myanmar kyat", "namibian dollar", 
    "nauru dollar", "nepalese rupee", "netherlands guilder", "new zealand dollar", "nicaraguan córdoba",
    "west african CFA franc", "nigerian naira", "north macedonian denar", "norwegian krone", 
    "oman rial", "pakistani rupee", "palauan dollar", "panamanian balboa", "papua new guinea kina", 
    "paraguayan guarani", "peruvian nuevo sol", "philippine peso", "polish złoty", "portuguese escudo",
    "qatari riyal", "romanian leu", "russian ruble", "rwandan franc", "east caribbean dollar", 
    "saint lucian dollar", "saint vincent and the grenadines dollar", "samoan tala", "san marino lira", 
    "sao tome and principe dobra", "saudi riyal", "senegalese franc", "serbian dinar", "seychellois rupee", 
    "sierra leonean leone", "singapore dollar", "slovak koruna", "slovenian tolar", "solomon islands dollar", 
    "somali shilling", "south african rand", "south sudanese pound", "spanish peseta", "sri lankan rupee", 
    "sudanese pound", "surinamese dollar", "swedish krona", "swiss franc", "syrian pound", "taiwan dollar", 
    "tajikistani somoni", "tanzanian shilling", "thai baht", "timor-lestean escudo", "west african CFA franc",
    "tongan paʻanga", "trinidad and tobago dollar", "tunisian dinar", "turkish lira", "turkmenistani manat",
    "tuvaluan dollar", "ugandan shilling", "ukrainian hryvnia", "united arab emirates dirham", "british pound", 
    "united states dollar", "uruguayan peso", "uzbekistani soʻm", "vanuatu vatu", "vatican lira", 
    "venezuelan bolívar", "vietnamese đồng", "yemeni rial", "zambian kwacha", "zimbabwean dollar", "palestinian pound"
]

currencyCodes = [
    "afn", "all", "dzd", "eur", "aoa", "ecd", "ars", "amd", "aud", "ats", "azn", "bsd", "bhd", 
    "bdt", "bbd", "byr", "eur", "bzd", "xaf", "btn", "bob", "bam", "bwp", "brl", "bnd", "bgn", 
    "bf", "bdi", "cve", "khm", "xaf", "clp", "cny", "cop", "kmf", "cdf", "cdf", "crc", "hrk", "cuc", 
    "cyp", "czk", "dkk", "djf", "ecd", "dop", "usd", "egp", "svc", "gnf", "ern", "est", "szl", "etb", 
    "fjd", "eur", "gfr", "gmd", "gel", "deu", "ghs", "eur", "grd", "bbd", "gtq", "gnf", "gbs", "gyd", 
    "htg", "hnl", "huf", "isk", "inr", "idr", "irr", "iqd", "iep", "ils", "itl", "jmd", "jpy", "jod", 
    "kzt", "kes", "aud", "kpw", "krw", "kwd", "kgs", "lak", "lvl", "lbp", "lsl", "lrd", "lyd", "chf", 
    "ltl", "luf", "mga", "mwk", "myr", "mvr", "mli", "mtd", "mid", "mxn", "fjd", "mdl", "mcf", "mnt", 
    "mne", "mad", "mzn", "mmk", "nad", "aud", "npr", "guilder", "nzd", "nic", "west african CFA franc", 
    "ngn", "mkd", "nok", "omr", "pkr", "usd", "pyg", "pen", "php", "pln", "ptc", "qar", "ron", "rub", 
    "rwf", "ecd", "usd", "sll", "sgd", "skk", "sit", "sol", "so", "zar", "ssp", "esp", "lkr", "sdg", 
    "srd", "sek", "chf", "syp", "twd", "tjs", "tzs", "thb", "timor", "xtl", "top", "ttd", "tnd", "try", 
    "tmt", "aud", "ugx", "uah", "aed", "gbp", "usd", "uyu", "uzs", "vuv", "vet", "vnd", "yer", "zmw", 
    "zwd", "ils"
]

#iterate through local database of VNR reports, read text and append to docs list
for file in os.listdir(directory):
    filePath = os.path.join(directory, file)

    if filePath.endswith('.pdf'):
        print(filePath)
        
        with pymupdf.open(filePath) as doc:
            pages = []
            
            for page in doc:
                pages.append(re.sub(r'\s+', ' ', page.get_text()))
            
            docs.append(" ".join(pages))
            
print('Tokenize Start')

#tokenize, remove certain parts of speech
tokenizer = RegexpTokenizer(r'\w+')
for index in range(len(docs)):
    docs[index] = docs[index].lower()
    tokens = tokenizer.tokenize(docs[index])
    tagged_tokens = pos_tag(tokens)
    
    filtered_tokens = []
    
    for word, tag in tagged_tokens:
        if tag not in ['NNP', 'NNPS']:
            filtered_tokens.append(word)
    
    docs[index] = filtered_tokens

print('Tokenize End')

print('Preprocess Start')

#preprocess to remove country specific words
docs = [[token for token in doc if token not in stopwords.words('english')] for doc in docs]
docs = [[token for token in doc if token not in countries] for doc in docs]
docs = [[token for token in doc if token not in nationality] for doc in docs]
docs = [[token for token in doc if token not in regions] for doc in docs]
docs = [[token for token in doc if token not in currencies] for doc in docs]
docs = [[token for token in doc if token not in currencyCodes] for doc in docs]
docs = [[token for token in doc if not token.isnumeric()] for doc in docs]
docs = [[token for token in doc if len(token) > 3] for doc in docs]

bigram = Phrases(docs, min_count = 10)

for index in range(len(docs)):
    for token in bigram[docs[index]]:
        if '_' in token:
            docs[index].append(token)

print('Preprocess End')
print(docs)

dictionary = Dictionary(docs)
dictionary.filter_extremes(no_below=5)
corpus = [dictionary.doc2bow(doc) for doc in docs]

dfCounts = dictionary.dfs

boostedCorpus = [
    [(token_id, count * dfCounts[token_id]) for token_id, count in doc_bow]
    for doc_bow in corpus
]

# print("\nCorpus inspection:")
# for i, doc in enumerate(corpus[:3]):  # check first few
#     print(f"Doc {i}: {doc}")
# 
# print("\nDictionary inspection:")
# print(f"Number of tokens in dictionary: {len(dictionary)}")
# print(f"Sample tokens: {list(dictionary.token2id.items())[:10]}")
# 
# print("\nFull corpus length:", len(corpus))
# print("Non-empty docs in corpus:", sum(1 for doc in corpus if len(doc) > 0))
# print("Empty docs:", [i for i, doc in enumerate(corpus) if not doc])
# 
# try:
#     print("\nSanity check: Training on first 3 documents only")
#     mini_lda = LdaModel(
#         corpus=corpus[:3],
#         id2word=dictionary,
#         num_topics=2,
#         passes=10,
#         iterations=50,
#         eval_every=None
#     )
#     print("Sanity check successful — LDA trained")
# except Exception as e:
#     print("Sanity check failed:", e)
s

num_topics = 30
chunksize = 27
passes = 50 #The number of model training in the whole corpus
iterations = 100 #Number of iterations of per document
eval_every = 1  # Don't evaluate model perplexity, takes too much time.

print('Training Start')

id2word = dictionary.id2token
lda = LdaModel(
    corpus=boostedCorpus,
    id2word=id2word,
    chunksize=chunksize,
    alpha=0.1,
    eta='auto',
    iterations=iterations,
    num_topics=num_topics,
    passes=passes,
    eval_every=eval_every
)

for i,topic in lda.print_topics(30):
    print(f'Top 10 words for topic #{i+1}:')
    print(topic)
    print('\n')
    
    
#return the top topics found in the corpus. Each topic is usually represented as a collection of words along with their respective weights
top_topics = lda.top_topics(corpus) 

#Topic Coherence measures the clarity and meaningfulness of these topics
avg_topic_coherence = sum([t[1] for t in top_topics]) / num_topics
print('Average topic coherence: %.4f.' % avg_topic_coherence)

from pprint import pprint
pprint(top_topics)