from flask import Flask, render_template, request
from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession
from pyspark.sql.types import StringType
from sparktorch import PysparkPipelineWrapper

from data_processing import clean_text, to_vector

app = Flask(__name__, template_folder='templates')


def load_PySpark():
    return SparkSession.builder \
        .appName("Flask APP") \
        .master("local[4]") \
        .config('spark.executor.memory', '50G') \
        .config("spark.driver.memory", "50G") \
        .config("spark.sql.analyzer.maxIterations", "6000") \
        .config("spark.driver.cores", "10") \
        .config("spark.driver.maxResultSize", "10G") \
        .config("spark.kryoserializer.buffer.max", "2000M") \
        .config("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp_2.12:3.4.4") \
        .config("spark.sql.autoBroadcastJoinThreshold", "-1") \
        .getOrCreate()


spark = load_PySpark()


def process_sentence(sentence):
    sentence_cleaned = clean_text(sentence)
    df_sentence = spark.createDataFrame([sentence_cleaned], schema=StringType()).toDF("text")
    return to_vector(df_sentence)


def predict(df):
    return PysparkPipelineWrapper \
        .unwrap(PipelineModel.load("model")) \
        .transform(df) \
        .select('predictions')


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def post_sentence():
    try:
        if request.method == 'POST':
            sentence = request.form.to_dict().get("text")
            df_processed = process_sentence(sentence)
            prediction = predict(df_processed)
            if prediction.collect()[0][0] == 1.0:
                return render_template("index.html", prediction="Yes, it is")
            else:
                return render_template("index.html", prediction="No, it's not")
    except:
        return render_template("index.html", prediction="Please, re-insert the sentence.")


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
    print("Open browser on address http://localhost:5000")
