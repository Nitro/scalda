
#scaLDA - Nitro's Open Source Topic Modeling Library

Welcome to scaLDA! This library allows you to train your very own [Latent Dirichlet Allocation (LDA)](https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation) topic models in Scala and Spark.  Specifically, this library is an implementation of the Online LDA algorithm presented in [_Online Learning for Latent Dirichlet Allocation_, Hoffman et al.](https://www.cs.princeton.edu/~blei/papers/HoffmanBleiBach2010b.pdf).   

Things you can do with scaLDA:

* Train an LDA model locally or in a distributed fashion using Spark.
* Given a learned topic model, infer the topic proportions within a given document.
* Evaluate a topic model with perplexity as well as semantic word coherence techniques.
* Compute the similarity between two documents by evaluating the similarity between their respective topic proportion distributions.

Check out some examples of how you can use scaLDA in this repo's [examples section](https://github.com/Nitro/scalda/blob/master/src/main/scala/com/nitro/scalda/examples/).

##Examples

###Train an LDA model locally
To train an LDA model locally you need two things

* An iterator over minibatches of documents.  A document is simply a ```String``` of the document contents.  A minibatch of documents is therefore an ```IndexedSeq[String]``` where the size of the minibatch is choosen by the user.  Therefore an iterator over minibatches is a ```Iterator[IndexedSeq[String]]```.   How this iterator is created depends on the particular way your documents are stored (i.e. local file system, S3, etc.) therefore it is up to the user to provide this iterator.
* An ```OnlineLDAParams``` object containing the parameters for the LDA model that you are going to train.

The following is an example taken from the [examples section](https://github.com/Nitro/scalda/blob/master/src/main/scala/com/nitro/scalda/examples/) in this repo.It creates an iterator over minibatches of documents from the NIPS corpus in this repo's [data sets section](https://github.com/Nitro/scalda/tree/master/datasets).

NOTE: Assume the following imports in the code below:
```scala
import scala.io._
import java.io._
```

First, we create `TextFileIterator`, which is an `Iterator` sub-class that provides minibatches from text files in a local directory.
```scala
class TextFileIterator(
  corpusDirectory: File, 
  mbSize: Int
) extends Iterator[IndexedSeq[String]] {

  private[this] val fileMinibatches = 
    Option(corpusDirectory.listFiles())
      .map(_.toSeq)
      .getOrElse(Seq.empty[File])
      .filter { f => !f.getName.startsWith(".") }
      .grouped(mbSize)

  override def hasNext = 
    fileMinibatches.hasNext

  override def next() = {
    println("processing next minibatch...")
    val nextMb = fileMinibatches.next()
    val stringMb = nextMb.map { f => 
      Source
        .fromFile(f, "ISO-8859-1")
        .getLines
        .mkString("\n")
    }
    stringMb.toIndexedSeq
  }
}

object LocalOnlineLDAExample extends App {

  val corpusDirectory = new File("datasets/nips_corpus")
  val vocabFile = new File("datasets/nips_vocab.txt")
  val mbSize = 100
  val numTopics = 20
  val numDocs = 6000
  
  val documents = new TextFileIterator(corpusDirectory, mbSize)
  val vocab = Source.fromFile(vocabFile).getLines.toSeq

  val p = OnlineLDAParams(
    vocabulary = vocab,
    alpha = 1.0 / numTopics,
    eta = 1.0 / numTopics,
    decay = 1024,
    learningRate = 0.7,
    maxIter = 100,
    convergenceThreshold = 0.001,
    numTopics = numTopics,
    totalDocs = numDocs,
    perplexity = true
  )
  
  //create an LDA instance with the given parameters
  val lda = LocalOnlineLDA(p)

  //train the model with the given minibatch iterator.
  val ldaModel = lda.inference(documents)
}
```


###Train an LDA Model with Spark
You can train an LDA model with Spark in an analogous way.  The two things you need here are

* An iterator over RDDs of documents.  Documents are again treated as ```String```'s.  However, this time a minibatch is represented by an ``RDD[string]`` so that we can perform operations on minibatches in parallel.  
* The exact same ```OnlineLDAParams``` object as the local version.

Here is an example implementation from the [examples section](https://github.com/Nitro/scalda/blob/master/src/main/scala/com/nitro/scalda/examples/).  In this particular example, the ```RDD``` minibatch iterator is created from documents in a directory within a local filesystem.  You will have to create your own custom iterator depending on where your documents are stored (e.g. HDFS, S3, etc.).  Also, training an LDA model with Spark requires an implicit Spark context.

```scala
class textFileRDDIterator(corpusDirectory: String, mbSize: Int)(implicit sc: SparkContext) extends Iterator[RDD[String]] {

  val directoryFile = new File(corpusDirectory)

  val fileMinibatches = directoryFile
    .listFiles()
    .grouped(mbSize)

  def hasNext = fileMinibatches.hasNext

  def next() = {

    val nextMb = fileMinibatches.next()
    val stringMb = nextMb.map(f => scala.io.Source.fromFile(f, "ISO-8859-1").getLines.mkString)

    sc.parallelize(stringMb)
  }

}

object DistributedOnlineLDAExample extends App {

  val corpusDirectory = args(0)
  val vocabFile = args(1)
  val mbSize = args(2).toInt
  val numTopics = args(3).toInt
  val numDocs = args(4).toInt

  val conf = new SparkConf()
    .setAppName("Distributed Online LDA Example")
    .setMaster("local[3]")

  implicit val sc = new SparkContext(conf)

  val myIter = new textFileRDDIterator(corpusDirectory, mbSize)

  val vocab = scala.io.Source.fromFile(vocabFile).getLines.toList

  val p = OnlineLDAParams(
    vocabulary = vocab,
    alpha = 1.0 / numTopics,
    eta = 1.0 / numTopics,
    decay = 1024,
    learningRate = 0.7,
    maxIter = 100,
    convergenceThreshold = 0.001,
    numTopics = numTopics,
    totalDocs = numDocs)


  val lda = new DistributedOnlineLDA(p)

  val trainedModel = lda.inference(myIter)

  lda.printTopics(trainedModel)

  sc.stop()
}
```


###Infer Topic Proportions for a Document
Once you have trained your LDA model, you might want to infer the proportions of the learned topics within a given document.  This is a great way to learn the 'concepts' and 'themes' that are present in a document based on its high probability topics.

The following example loads a previous learned and serialized model and uses it to infer the topic proportions for a given document.

```scala
object TopicProportionsExample extends App {

  val modelLocation = args(0)
  val docLocation = args(1)

  val testDoc = Source.fromFile(docLocation).getLines().mkString

  val lda = LocalOnlineLDA()

  val myModelTry = lda.loadModel(modelLocation)

  val topicProps = lda.topicProportions(testDoc, myModelTry.get)

}

```
