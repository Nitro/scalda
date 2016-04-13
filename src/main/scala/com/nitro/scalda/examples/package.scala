package com.nitro.scalda

import java.io.File

import scala.io.Source

package object examples {

  val lines: File => Iterator[String] =
    f => Source.fromFile(f).getLines()

  val text: File => String =
    lines andThen { _.mkString(" ") }

  def log(message: => String, on: Boolean = true): Unit =
    if (on)
      System.err.println(message)
    else
      ()

  def getOrElse(args: Array[String])(index: Int, alt: => String): String =
    Option(args(index)).getOrElse(alt)

  def getOrElse[T](
    args: Array[String],
    index: Int,
    alt: => T,
    convert: String => T
  ): T =
    Option(args(index))
      .map(convert)
      .getOrElse(alt)

}
