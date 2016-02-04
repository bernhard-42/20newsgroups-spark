
## 1 Download Data and load into HDFS


```bash
%sh

cd /tmp
mkdir 20newsgroups
cd 20newsgroups

wget http://qwone.com/~jason/20Newsgroups/20news-bydate.tar.gz

tar -zxf 20news-bydate.tar.gz

ls  /tmp/20newsgroups/20news-bydate-train
```


```bash
%sh

hdfs dfs -put /tmp/20newsgroups 

hdfs dfs -ls /user/zeppelin/20newsgroups/20news-bydate-train
```


## 2 Multinomial Logistic Regression with MLLib (Scala)


### 2.1 Prepare Training and Test Data 


```scala
%spark

import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.feature.{HashingTF, IDF}
import scala.collection.immutable.HashMap
import org.apache.spark.mllib.regression.LabeledPoint
import scala.math.pow

val categories = Array("alt.atheism", "soc.religion.christian", "comp.graphics", "sci.med")
val categoryMap = categories.zipWithIndex.toMap

val numFeatures = pow(2,18).toInt  // default is 2**20, so reduce on smaller machines 

def tokenize(line: String): Array[String] = {
    line.split("""\W+""").map(_.toLowerCase)
}

def prepareData(typ: String) = {
    categories.map(category => {
        val wordsData = sc.wholeTextFiles("/user/zeppelin/20newsgroups/20news-bydate-" + typ + "/" + category)
                          .map(message => tokenize(message._2).toSeq)

        val hashingTF = new HashingTF(pow(2,18).toInt)
        val featuredData = hashingTF.transform(wordsData).cache()

        val idf = new IDF().fit(featuredData)
        val tfidf = idf.transform(featuredData)
        tfidf.map(row => LabeledPoint(categoryMap(category),row))
    }).reduce(_ union _)
}

val twenty_train = prepareData("train").cache()
val twenty_test  = prepareData("test").cache()

```


### 2.2 Create Model


```scala
%spark

import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS

val model = new LogisticRegressionWithLBFGS().setNumClasses(4).run(twenty_train)

```


### 2.3 Validate Model


```scala
%spark

import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.evaluation.MulticlassMetrics

def validate(predictionsAndLabels: RDD[(Double, Double)], labels: Array[String]) = {
    val metrics = new MulticlassMetrics(predictionsAndLabels)

    println("")
    println("CONFUSION MATRIX")
    println(metrics.confusionMatrix)
    println("")
    println("CATEGORY                 PRECISION  RECALL")
    
    for (i <- 0 to labels.size-1) {
        val l = labels(i)
        val p = metrics.precision(i)
        val r = metrics.recall(i)
        println(f"$l%22s:  $p%2.3f      $r%2.3f")
    }
    println("")
}

```


```scala
%spark


val predictionsAndLabels = twenty_test.map { case LabeledPoint(label, features) =>
  val prediction = model.predict(features)
  (prediction, label)
}

validate(predictionsAndLabels, categories)
val metrics = new MulticlassMetrics(predictionsAndLabels)
val precision = metrics.precision

```


## 3 Naive Bayes Classification with Spark ML Pipeline (Scala)


### 3.1 Prepare Training and Test Data


```scala
%spark

val categories = Array("alt.atheism", "soc.religion.christian", "comp.graphics", "sci.med")

def prepareDF(typ: String) = {
    val rdds = categories.map(category => sc.wholeTextFiles("/user/zeppelin/20newsgroups/20news-bydate-" + typ + "/" + category)
                                            .map(msg => (category, msg._2)))
    sc.union(rdds).toDF("category", "message")
}

val twenty_train_df = prepareDF("train").cache()
val twenty_test_df  = prepareDF("test").cache()

```


### 3.2 Create Pipeline and Model

`Note: As of Spark 1.6.0, Naive Bayes for Spark ML is still "experimental"`




```scala
%spark

import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer, StringIndexer, IndexToString}
import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.Pipeline

val indexer   = new StringIndexer().setInputCol("category").setOutputCol("label").fit(twenty_train_df)

val tokenizer = new Tokenizer().setInputCol("message").setOutputCol("words")
val hashingTF = new HashingTF().setInputCol("words").setOutputCol("rawFeatures")
val idf       = new IDF().setInputCol("rawFeatures").setOutputCol("features")

val nb        = new NaiveBayes().setFeaturesCol("features").setLabelCol("label").setSmoothing(1.0).setModelType("multinomial") // implicit name of outputCol: "prediction"

val converter = new IndexToString().setInputCol("prediction").setOutputCol("predictedCategory").setLabels(indexer.labels)   // optional


val pipeline = new Pipeline().setStages(Array(indexer, tokenizer, hashingTF, idf, nb, converter))

val model = pipeline.fit(twenty_train_df)

```


### 3.3 Validation


```scala
%spark

import org.apache.spark.sql.Row
import org.apache.spark.mllib.evaluation.MulticlassMetrics

val validation = model.transform(twenty_test_df)

val predictions = validation.select("label", "prediction")
val predictionsAndLabels = predictions.map {case Row(p: Double, l: Double) => (p, l)}

validate(predictionsAndLabels, indexer.labels)

```


## 4 Naive Bayes Classification with Spark ML Pipeline (Python)


### 4.1 Prepare Training and Test Data


```python
%pyspark

categories = ["alt.atheism", "soc.religion.christian", "comp.graphics", "sci.med"]

LabeledDocument = Row("category", "text")

def prepareDF(typ):
    rdds = [sc.wholeTextFiles("/user/zeppelin/20newsgroups/20news-bydate-" + typ + "/" + category)\
              .map(lambda x: LabeledDocument(x[0].split("/")[-2], x[1]))\
            for category in categories]
    return sc.union(rdds).toDF()


twenty_train_df = prepareDF("train").cache()
twenty_test_df  = prepareDF("test") .cache()

```


### 4.2 Create Pipeline and Model


```python
%pyspark

from pyspark.ml.feature import HashingTF, IDF, Tokenizer, StringIndexer, StringIndexerModel
from pyspark.ml.classification import NaiveBayes
from pyspark.ml import Pipeline

indexer   = StringIndexer(inputCol="category", outputCol="label").fit(twenty_train_df)
categories = indexer._call_java("labels")   # BUG in 1.5.2, indexer.labels note exposed

tokenizer = Tokenizer(inputCol="text", outputCol="words")
hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures")
idf       = IDF(inputCol="rawFeatures", outputCol="features")
nb        = NaiveBayes(featuresCol="features", labelCol="label", smoothing=1.0, modelType="multinomial")

pipeline = Pipeline(stages=[indexer, tokenizer, hashingTF, idf, nb])

model = pipeline.fit(twenty_train_df)

```


### 4.3 Validation


```python
%pyspark

from pyspark.mllib.evaluation import MulticlassMetrics

prediction = model.transform(twenty_test_df)

predictionAndLabels = prediction.select("label", "prediction").rdd.cache()
metrics = MulticlassMetrics(predictionAndLabels)

print "\nCONFUSION MATRIX"
print metrics.confusionMatrix().toArray()

print "\nCATEGORY                 PRECISION  RECALL  F1"
for i in range(len(categories)):
    l = categories[i]
    p = metrics.precision(float(i))
    r = metrics.recall(float(i))
    f = metrics.fMeasure(float(i))
    print "%22s:  %2.2f       %2.2f    %2.2f" % (l, p, r, f)
```

