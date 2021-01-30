import os
from pathlib import Path

current_path = Path(__file__).resolve().parent
DATA_PATH = os.path.join(current_path, './res/')
GENERATOR_PATH = os.path.join(current_path, './../generator/')
CONFIG_PATH = os.path.join(current_path, './../config/')
SAMPLE_PATH = os.path.join(current_path, './../component/sample/')
TRANSFORMATION_PATH = os.path.join(current_path, './../transformation/')

# mask settings
ORIGIN = 0
TASK_MASK = 1
MODIFIED_MASK = 2

VARIABLE = 'variable'
JSON = 'json'
CSV = 'csv'
OUT_FORMATS = [VARIABLE, JSON, CSV]


NLP_TASK_MAP = {
    'UT': 'Universal transform',
    'ABSA': 'Aspect Based Sentiment Analysis',
    'SA': 'Sentiment Analysis',
    'CWS': 'Chinese Word Segmentation',
    'NER': 'Named Entity Recognition',
    'POS': 'Part-of-Speech Tagging',
    'DP': 'Dependency Parsing',
    'MRC': 'Machine Reading Comprehension',
    'SM': 'Semantic Matching',
    'NLI': 'Natural language inference',
    'RE': 'Relation Extraction',
    'Coref': 'Coreference resolution'
}

TASK_TRANSFORMATION_PATH = dict((task, os.path.join(TRANSFORMATION_PATH, task)) for task in NLP_TASK_MAP)

# indicate allowed transformations of specific task
ALLOWED_TRANSFORMATIONS = {
    'UT': [
        'Case',
        'Keyboard',
        #'Number',
        'Ocr',
        'Spelling',
        #'Tense',
        'Typos',
        #'WordEmbedding',
        'WordNetAntonym',
        'WordNetSynonym'
    ],
    'ABSA': [

    ],
    'SA': [

    ],
    'CWS': [

    ],
    'NER': [

    ],
    'POS': [

    ],
    'DP': ['RemoveSubtree',
           'AddSubtree'
    ],
    'MRC': [

    ],
    'SM': [

    ],
    'NLI': [

    ],
    'RE': [

    ],
    'Coref': [

    ]
}

# -------------------------UT settings---------------------------
STOP_WORDS = ['I', 'my', 'My', 'mine', 'you', 'your', 'You', 'Your', 'He', 'he', 'him', 'Him',
              'His', 'his', 'She', 'she', 'her', 'Her', 'it', 'It', 'they', 'They', 'their',
              'Their', 'am', 'Are', 'are', 'Is', 'is', 'And', 'and', 'or', 'nor', 'A', 'a',
              'An', 'an', 'the', 'The', 'Have', 'have', 'Has', 'has', 'in', 'In', 'by', 'By',
              'on', 'On', 'of', 'Of', 'at', 'At', 'from', 'From']

# back translation model
TRANS_FROM_MODEL = "allenai/wmt16-en-de-dist-6-1"
TRANS_TO_MODEL = "allenai/wmt19-de-en-6-6-base"

# Offline Vocabulary
EMBEDDING_PATH = os.path.join(DATA_PATH, './word_embedding/sim_words_dic.json')
VERB_PATH = os.path.join(DATA_PATH, './word_tense/verb_tenses.json')

# Spelling error vocabulary
SPELLING_ERROR_DIC = os.path.join(DATA_PATH, 'word',  'spelling_en.txt')
EMBEDDING_PATH = os.path.join(DATA_PATH,'./word_embedding/sim_words_dic.json')
VERB_PATH = os.path.join(DATA_PATH, './word_tense/verb_tenses.json')
ADVERB_PATH = os.path.join(DATA_PATH,'./word_adverb/neu_adverb_word_228.txt')
TWITTER_PATH = os.path.join(DATA_PATH, './word_twitter/twitter_contraction.json')
SENT_PATH = os.path.join(DATA_PATH,'./add_sentence/test.txt')

# UT word_contraction
REVERSR_CONTRACTION_MAP = {
    "isn't": "is not", "ain't": "is not",
    "aren't": "are not", "are n't": "are not",
    "can't": "can not", "ca n't": "can not",
    "couldn't": "could not", "couldn't": "could not", "could n't": "could not",
    "didn't": "did not", "did n't": "did not",
    "doesn't": "does not", "does n't": "does not",
    "don't": "do not", "don ' t": "do not",
    "hadn't": "had not", "had n't": "had not",
    "hasn't": "has not", "has n't": "has not",
    "haven't": "have not", "have n't": "have not",
    "he'd": "he would", "he 'd": "he would",
    "he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y":
        "how do you", "how'll": "how will", "how's": "how is",
    "I'd": "I would", "I'll": "I will", "I'm": "I am",
    "I've": "I have", "i'd": "i would", "i'll": "i will",
    "i'm": "i am", "i've": "i have", "isn't": "is not",
    "it'd": "it would", "it'll": "it will", "it's": "it is", "ma'am":
        "madam", "might've": "might have", "mightn't": "might not",
    "must've": "must have", "mustn't": "must not", "needn't":
        "need not", "oughtn't": "ought not", "shan't": "shall not",
    "she'd": "she would", "she'll": "she will", "she's": "she is",
    "should've": "should have", "shouldn't": "should not", "that'd":
        "that would", "that's": "that is", "there'd": "there would",
    "there's": "there is", "they'd": "they would",
    "they'll": "they will", "they're": "they are",
    "they've": "they have", "wasn't": "was not", "we'd": "we would",
    "we'll": "we will", "we're": "we are", "we've": "we have",
    "weren't": "were not", "what're": "what are", "what's": "what is",
    "when's": "when is", "where'd": "where did", "where's": "where is",
    "where've": "where have", "who'll": "who will", "who's": "who is",
    "who've": "who have", "why's": "why is", "won't": "will not",
    "would've": "would have", "wouldn't": "would not",
    "you'd": "you would", "you'd've": "you would have",
    "you'll": "you will", "you're": "you are", "you've": "you have"
}

CONTRACTION_MAP = {
    'is not': "isn't", 'are not': "aren't", 'cannot': "can't",
    'could not': "couldn't", 'did not': "didn't", 'does not':
        "doesn't", 'do not': "don't", 'had not': "hadn't", 'has not':
        "hasn't", 'have not': "haven't", 'he is': "he's", 'how did':
        "how'd", 'how is': "how's", 'I would': "I'd", 'I will': "I'll",
    'I am': "I'm", 'i would': "i'd", 'i will': "i'll", 'i am': "i'm",
    'it would': "it'd", 'it will': "it'll", 'it is': "it's",
    'might not': "mightn't", 'must not': "mustn't", 'need not': "needn't",
    'ought not': "oughtn't", 'shall not': "shan't", 'she would': "she'd",
    'she will': "she'll", 'she is': "she's", 'should not': "shouldn't",
    'that would': "that'd", 'that is': "that's", 'there would':
        "there'd", 'there is': "there's", 'they would': "they'd",
    'they will': "they'll", 'they are': "they're", 'was not': "wasn't",
    'we would': "we'd", 'we will': "we'll", 'we are': "we're", 'were not':
        "weren't", 'what are': "what're", 'what is': "what's", 'when is':
        "when's", 'where did': "where'd", 'where is': "where's",
    'who will': "who'll", 'who is': "who's", 'who have': "who've", 'why is':
        "why's", 'will not': "won't", 'would not': "wouldn't", 'you would':
        "you'd", 'you will': "you'll", 'you are': "you're",
}

# UT sent_add_sent
MIN_SENT_TRANS_LENGTH = 10

# ---------------------------CWS settings---------------------------
CWS_DATA_PATH = os.path.join(current_path, './res/CWS_DATA/')
NUM_LIST = ['零', '一', '二', '三', '四', '五', '六', '七', '八', '九', '十', '百', '千', '万', '亿']
# Judge whether the number is less than ten
NUM_FLAG1 = 9
# Judge whether the number is less than ten thousand
NUM_FLAG2 = 12
# number begin
NUM_BEGIN = 1
# number end
NUM_END = 9
abbreviation_path = CWS_DATA_PATH + 'abbreviation'
NAME_PATH = CWS_DATA_PATH + '姓.txt'
WORD_LIST_PATH = CWS_DATA_PATH + 'dict'
ABAB_PATH = CWS_DATA_PATH + '60ABAB'
AABB_PATH = CWS_DATA_PATH + '650AABB'
AONEA_PATH = CWS_DATA_PATH + 'A-one-A'

# ---------------------- NLI & SM settings -----------------------------------
BLACK_LIST_WORD = ["here", "goodness", "yes", "no", "decision", "growing", "priority", "cheers", "volume", "right",
                   "left", "goods", "addition", "income", "indecision", "there", "parent", "being", "parents",
                   "lord", "lady", "put", "capital", "lowercase", "unions"]

# -------------------------SA settings---------------------------
SA_PERSON_PATH = os.path.join(current_path, './res/SAInfo/person_info.csv')
SA_MOVIE_PATH = os.path.join(current_path, './res/SAInfo/movie_info.csv')

SA_DOUBLE_DENIAL_DICT = {'poor': 'not good', 'bad': 'not great', 'lame': 'not interesting',
                         'awful': 'not awesome', 'great': 'not bad', 'good': 'not poor',
                         'applause': 'not discourage',
                         'recommend': "don't prevent", 'best': 'not worst', 'encourage': "don't discourage",
                         'entertain': "don't disapprove",
                         'wonderfully': 'not poorly', 'love': "don't hate",
                         'interesting': "not uninteresting", 'interested': 'not ignorant',
                         'glad': 'not reluctant', 'positive': 'not negative', 'perfect': 'not imperfect',
                         'entertaining': 'not uninteresting',
                         'moved': 'not moved', 'like': "don't refuse", 'worth': 'not undeserving',
                         'better': 'not worse', 'funny': 'not uninteresting', 'awesome': 'not ugly',
                         'impressed': 'not impressed'}

# -------------------------POS settings---------------------------
BERT_MODEL_NAME = 'bert-base-uncased'

# -------------------------ABSA settings---------------------------
NEGATIVE_WORDS_LIST = ['doesn\'t', 'don\'t', 'didn\'t', 'no', 'did not', 'do not',
                       'does not', 'not yet', 'not', 'none', 'no one', 'nobody', 'nothing',
                       'neither', 'nowhere', 'never', 'hardly', 'scarcely', 'barely']
DEGREE_WORD_LIST = ['absolutely', 'awfully', 'badly', 'barely', 'completely', 'decidedly',
                    'deeply', 'enormously', 'entirely', 'extremely', 'fairly', 'fully',
                    'greatly', 'highly', 'incredibly', 'indeed', 'very', 'really']
PHRASE_LIST = ['ASJP', 'ADVP', 'CONJP', 'FRAG', 'INTJ', 'LST', 'NAC',
               'NP', 'NX', 'PP', 'PRN', 'PRT', 'QP', 'RRC', 'UCP', 'VP',
               'WHADJP', 'WHAVP', 'WHNP', 'WHPP', 'X', 'S', 'SBAR']
ABSA_DATA_PATH = os.path.join(current_path, './res/ABSA_DATA/')
ABSA_CONSTITUENT_PATH = os.path.join(current_path, './res/ABSA_DATA/elmo-constituency-parser-2020.02.10.tar.gz')


# -------------------------NER settings---------------------------
NER_OOV_ENTITIES = os.path.join(DATA_PATH, './NER/OOVentities')
LONG_ENTITIES = os.path.join(DATA_PATH, './NER/label_count_for_length.json')
CROSS_ENTITIES = os.path.join(DATA_PATH, './NER/Multientity')

# ---------------------------DP settings---------------------------
WIKIDATA_STATEMENTS = ['P101', 'P19', 'P69', 'P800', 'P1066', 'P50', 'P57', 'P136', 'P921', 'P159', 'P740', 0]
CLAUSE_HEAD = {'P101': 'who worked at ', 'P19': 'who was born in ', 'P69': 'who was educated at ',
              'P800': 'whose notable work is ', 'P1066': 'who is the student of ', 'P136': 'which is a ',
              'P57': 'which is directed by ', 'P921': 'with the subject of the ', 'P50': 'which is written by ',
              'P159': 'headquartered in ', 'P740': 'established in ', 0: 'which is a '}
WIKIDATA_INSTANCE = {'instance': 'P31',
                     'disambiguation': 'Q4167410',
                     'human': 'Q5'}
