organization := "com.gonitro.research"

name := "topic_models"

version := "0.7.1"

lazy val jvmVer = com.nitro.build.Runtime.Jvm7


scalaVersion := "2.10.5"

crossScalaVersions := Seq("2.10.5", "2.11.6")
javaOptions ++= Seq("-Xmx9G", "-Xms256M")

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


