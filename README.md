
#scaLDA - Nitro's Open Source Topic Modeling Library

Welcome to scaLDA! This library allows you to train your very own [Latent Dirichlet Allocation (LDA)](https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation) topic models in Scala and Spark.  Specifically, this library is an implementation of the Online LDA algorithm presented in [_Online Learning for Latent Dirichlet Allocation_, Hoffman et al.](https://www.cs.princeton.edu/~blei/papers/HoffmanBleiBach2010b.pdf).   

Things you can do with scaLDA:

* Train an LDA model locally or in a distributed fashion using Spark.
* Given a learned topic model, infer the topic proportions within a given document.
* Evaluate a topic model with perplexity as well as semantic word coherence techniques.
* Compute the similarity between two documents by evaluating the similarity between their respective topic proportion distributions.

Check out some examples of how you can use scaLDA in this repo's [examples section](https://github.com/Nitro/scalda/blob/master/src/main/scala/com/nitro/scalda/examples/).

##Examples

Assume the following imports and definitions are in scope for *all examples*:
```scala
import scala.io.File
import java.io.Source

val lines: File => Iterator[String] =
  f => Source.fromFile(f).getLines()

val text: File => String =
  lines andThen { _.mkString(" ") }
```

All of the example code here is taken from the [examples section](https://github.com/Nitro/scalda/blob/master/src/main/scala/com/nitro/scalda/examples/) in this repo.It creates an iterator over minibatches of documents from the NIPS corpus in this repo's [data sets section](https://github.com/Nitro/scalda/tree/master/datasets).


###Train an LDA model locally
To train an LDA model locally you need:

1. An iterator over minibatches of documents.  A document is simply a ```String``` of the document contents.  A minibatch of documents is therefore an ```IndexedSeq[String]``` where the size of the minibatch is choosen by the user.  Therefore an iterator over minibatches is a ```Iterator[IndexedSeq[String]]```.   How this iterator is created depends on the particular way your documents are stored (i.e. local file system, S3, etc.) therefore it is up to the user to provide this iterator.
2. An ```OnlineLDAParams``` object containing the parameters for the LDA model that you are going to train.


Learning topics using Online LDA on a local file system:
```scala
object LocalOnlineLDAExample extends App {

  val corpusDirectory = new File("datasets/nips_corpus")
  val vocabFile = new File("datasets/nips_vocab.txt")
  val mbSize = 100
  val numTopics = 20
  val numDocs = 6000
  
  val lda = LocalOnlineLda(
    OnlineLdaParams(
      vocabulary = lines(vocabFile).toIndexedSeq,
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
  )
  
  val model = lda.inference(new TextFileIterator(corpusDirectory, mbSize))
  
  lda.printTopics(model)
}

class TextFileIterator(
    corpusDirectory: File,
    mbSize: Int,
    logging: Boolean = true
) extends Iterator[IndexedSeq[String]] {

  // mutable state
  private[this] val fileMinibatches =
    Option(corpusDirectory.listFiles())
      .map(_.toSeq)
      .getOrElse(Seq.empty[File])
      .filter { f => !f.getName.startsWith(".") }
      .grouped(mbSize)

  override def hasNext =
    fileMinibatches.hasNext

  override def next() =
    fileMinibatches
      .next()
      .map { text }
      .toIndexedSeq
}
```


###Train an LDA Model with Spark
You can train an LDA model with Spark in an analogous way. You need:

1. An iterator over RDDs of documents.  Documents are again treated as ```String```'s.  However, this time a minibatch is represented by an ``RDD[string]`` so that we can perform operations on minibatches in parallel.  
2. The exact same ```OnlineLdaParams``` object as the local version.

In this particular example, the ```RDD``` minibatch iterator is created from documents in a directory within a local filesystem.  You will have to create your own custom iterator depending on where your documents are stored (e.g. HDFS, S3, etc.).  Also, training an LDA model with Spark requires an implicit Spark context.

```scala
object DistributedOnlineLDAExample extends App {

  val corpusLoc = new File(args(0))
  val vocabFile = new File(args(1))
  val mbSize = args(2).toInt
  val numTopics = args(3).toInt
  val numDocs = args(4).toInt

  implicit val sc = new SparkContext(
    new SparkConf()
      .setAppName("Distributed Online LDA Example")
      .setMaster("local[3]")
  )
  
  val lda = new DistributedOnlineLda(
    OnlineLdaParams(
      vocabulary = lines(vocabFile).toIndexedSeq,
      alpha = 1.0 / numTopics,
      eta = 1.0 / numTopics,
      decay = 1024,
      learningRate = 0.7,
      maxIter = 100,
      convergenceThreshold = 0.001,
      numTopics = numTopics,
      totalDocs = numDocs
    )
  )

  val model = lda.inference(new TextFileRddIterator(corpusLoc, mbSize))

  lda.printTopics(model)

  sc.stop()
}

class TextFileRddIterator(
    corpusDirectory: File,
    mbSize: Int,
    logging: Boolean = true
)(implicit sc: SparkContext) extends Iterator[RDD[String]] {

  // mutable state
  private[this] val fileMinibatches =
    Option(corpusDirectory.listFiles)
      .map(_.toSeq)
      .getOrElse(Seq.empty[File])
      .grouped(mbSize)

  override def hasNext =
    fileMinibatches.hasNext

  override def next() = {
    log("[RDD] next minibatch", on = logging)
    sc.parallelize {
      for {
        f <- fileMinibatches.next()
      } yield text(f)
    }
  }
}
```


###Infer Topic Proportions for a Document
Once you have trained your LDA model, you might want to infer the proportions of the learned topics within a given document.  This is a great way to learn the 'concepts' and 'themes' that are present in a document based on its high probability topics.

The following example loads a previous learned and serialized model and uses it to infer the topic proportions for a given document.

```scala
object TopicProportionsExample extends App {

  val modelLocation = new File(args(0))
  val docLocation = new File(args(1))  
  
  val testDoc = text(docLocation)
 
  val lda = LocalOnlineLda.empty
  val model = lda.loadModel(modelLocation).get
  
  val topicProps = lda.topicProportions(
    testDoc,
    model,
    Some(com.nitro.scalda.tokenizer.StanfordLemmatizer())
  )
  
  println(
    topicProps
      .toSeq
      .sortBy { case (prob, _) => -prob }
      .map {
        case (prob, topicIdx) => s"$prob,$topicIdx"
      }
      .mkString("\n")
  )
}
```

###Compute Document Similarity Example
Once you have trained your LDA model, you might want to use inferred topic distributions to compute a pairwise document similarity. The following example does this using the topics of a learned LDA model as the vector space, inferred proportions as the document's vector, and the cosine similarity metric.

```scala
object TopicProportionsExample extends App {
  
  val doc1Location = new File(args(0)) 
  val doc2Location = new File(args(1))
  val modelLocation = new File(args(2))
  val vocabFile = new File(args(3))
  val numTopics = args(4).toInt
  val numDocs = args(5).toInt

  val lda = LocalOnlineLda(
    OnlineLdaParams(
      lines(vocabFile).toIndexedSeq,
      1.0 / numTopics,
      1.0 / numTopics,
      1024,
      0.7,
      100,
      0.001,
      numTopics,
      numDocs
    )
  )

  val model = lda.loadModel(modelLocation).get

  val docSim12 = lda.documentSimilarity(
    text(doc1Location),
    text(doc2Location),
    model
  )
  
  println(docSim12)
}
```

###Compute Top Topic Word Coherence 
Once you have trained your LDA model, you often want to find out whether or not the learned topics are useful and coherent.

One technique is to look at the top words (by probability) for each topic and compute the average cosine similarity of all of a topic's top word vectors.

```scala
object WordVectorCoherenceExample extends App {

  val modelLoc = new File(args(0))
  val corpusLoc = new File(args(1))

  val model = LocalOnlineLda.empty.loadModel(modelLoc).get

  implicit val sc = new SparkContext(
    new SparkConf()
      .setAppName("Word Vector Coherence Example")
      .setMaster("local[3]")
  )

  val docRDD = sc
    .parallelize {
      Option(corpusLoc.listFiles())
        .map { _.toSeq }
        .getOrElse(Seq.empty[File])
    }
    .map { text }
    
  import com.nitro.scalda.evaluation.coherence

  val wordVecs = coherence.SparkWord2Vec.learnWordVectors(docRDD)

  val topicCoherence = coherence.WordVectorCoherence.getTopicCoherence(
    model, 
    wordVecs
  )

  topicCoherence.foreach { println }

  sc.stop()
}
```
