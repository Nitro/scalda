package com.nitro.scalda.examples

import java.io.File

/** WARNING: Contains mutable state. */
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

  override def next() = {
    log("[Local FS] next minibatch...", on = logging)
    fileMinibatches
      .next()
      .map { text }
      .toIndexedSeq
  }
}