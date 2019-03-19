import java.io.FileReader
import java.io.FileNotFoundException
import java.io.IOException

class Demo(name: String, age: Int){
	println(s"$name $age")
}

object Demo{
	def main(args: Array[String]){
		println("Hello world")
		delay(time())
		printStr("chinese","jiaozi","delicious")
		println(add())
		println(conv(cast,x=>x*x,5,"6"))
		new Demo("me", 18)

		try {
			var f = new FileReader("123.txt")
		} catch {
			case ex:FileNotFoundException => {
				println("Missing file exception")
			}
			case ex:IOException => {
				println("IO Exception")
			}
			case _: Throwable => {
				println("123 456")
			}
		} finally {
			println("Exiting finally...")
		}
	}

	def add(x: Int = 3 , y: Int = 5) = x + y

	def conv(f: String=> Int, g: Int=> Int, v: Int, d: String) = f(d) * g(v)

	def cast(x: String) = x.toInt

	def time() = {
      println("Getting time in nano seconds")
      System.nanoTime
    }

    def delay(t: => Long){
      println("In delayed method")
      println("Param: " + t)
    }

    def printStr(strs: String *){
    	strs.foreach(println)
    }

    def show(x: Option[String]) = x match {
    	case Some(s) => s
    	case None => "?"
    	case _ => ""
    }

    @throws(classOf[NumberFormatException])
    def validate(){  
        "abc".toInt  
    }  

    class Complex(real: Double, imaginary: Double){
    	val re = real
    	val im = imaginary
    }

    abstract class Tree
	case class Sum(l: Tree, r: Tree) extends Tree
	case class Var(n: String) extends Tree
	case class Const(v: Int) extends Tree

	type Environment = String => Int

	class Refer[T](x: T){
		private var content: T = x
		private var name: String = null
		def set(value: T) {content = value}
		def get = content

		private def print = println

		def this(x: T, y: String){
			this(x)
			this.name = y
		}
	}

	class ExceptionExample2{  
	    def validate(age:Int)={  
	        if(age<18)  
	            throw new ArithmeticException("You are not eligible")  
	        else println("You are eligible")  
	    }  
	}  
}