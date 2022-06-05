# useful imports
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
from pyspark.ml import PipelineModel
from pyspark.ml.functions import vector_to_array
from pyspark.sql.functions import *

text_processor = TextPreProcessor(
    normalize=['url', 'email', 'percent', 'money', 'phone', 'user',
               'time', 'date', 'number'],
    annotate={"hashtag", "allcaps", "elongated", "repeated",
              'emphasis', 'censored'},
    fix_html=True,
    segmenter="twitter",
    corrector="twitter",
    unpack_hashtags=True,
    unpack_contractions=True,
    spell_correct_elong=False,
    tokenizer=SocialTokenizer(lowercase=True).tokenize,
    dicts=[emoticons]
)


@udf
def tweet_processor(s):
    return " ".join(text_processor.pre_process_doc(s))


punctuation = "!$%&'()*+, -./:;<=>?[\]^_`{|}~«»"


def clean_text(sentence):
    sentence = sentence.lower() \
        .replace(r'http\S+', '') \
        .replace("\n", " ") \
        .replace(r'[0-9]{5,}', "")

    for p in punctuation:
        sentence = sentence.replace(p, " ")

    sentence = sentence.replace(" +", " ") \
        .strip()

    return " ".join(text_processor.pre_process_doc(sentence))


def to_vector(df):
    return PipelineModel.load("pipeline") \
        .transform(df) \
        .select(explode(col("finished_sentence_embeddings")).alias('features')) \
        .withColumn("features", vector_to_array("features")) \
        .select([expr('features[' + str(x) + ']') for x in range(100)])
