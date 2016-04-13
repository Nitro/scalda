package com.nitro.scalda.tokenizer

trait Tokenizer[T] {

  def tokenize(text: String): Seq[T]

}

