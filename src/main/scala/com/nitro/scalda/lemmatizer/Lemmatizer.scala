package com.nitro.scalda.lemmatizer

trait Lemmatizer[T] {

  def lemmatize(rawText: String): Seq[T]

}
