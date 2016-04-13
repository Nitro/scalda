import com.typesafe.sbt.SbtScalariform

organization := "com.gonitro"
name         := "topic_models"
version      := "0.8.0"

scalaVersion       := "2.10.6"
crossScalaVersions := Seq("2.10.6", "2.11.8")
javaOptions        ++= Seq("-Xmx9G", "-Xms256M")

libraryDependencies ++= Seq(
  ("org.apache.spark" %% "spark-core" % "1.3.0").
    exclude("org.mortbay.jetty", "servlet-api").
    exclude("commons-beanutils", "commons-beanutils-core").
    exclude("commons-collections", "commons-collections").
    exclude("commons-logging", "commons-logging").
    exclude("com.esotericsoftware.minlog", "minlog")
  ,
  "org.apache.spark" %% "spark-mllib" % "1.3.0",
  "org.scalatest" %% "scalatest" % "2.2.4" % Test,
  "edu.stanford.nlp" % "stanford-corenlp" % "1.3.4" artifacts(Artifact("stanford-corenlp", "models"), Artifact("stanford-corenlp"))

)

SbtScalariform.defaultScalariformSettings

