import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.sql.SQLContext
import scala.io.Source
import java.io._
import org.apache.spark.ml.feature.Word2Vec
import org.apache.spark.ml.feature.Word2VecModel
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.classification.LogisticRegressionModel
import scala.collection.mutable.Set
object SimpleApp {

	def subdirs(dir: File): Iterator[File] = {  
		val d = dir.listFiles.filter(_.isDirectory)  
		val f = dir.listFiles.toIterator  
		f ++ d.toIterator.flatMap(subdirs _) 
		return f
    }  
	def return_bookname(path: String):String = {
		val bookname_raw = path.split("\\\\")
		val bookname_txt = bookname_raw(bookname_raw.length-1)
		val bookname_txt_array = bookname_txt.split('-')
		val bookname = bookname_txt_array(2)
		val book_name_real_array = bookname.split('.')
		val book_name_real = book_name_real_array(0)
		return book_name_real
	}
	
	def return_author(path: String):String = {
		val author_raw = path.split("\\\\")
		val author_txt = author_raw(author_raw.length-1)
		val author_txt_array = author_txt.split('-')
		val author = author_txt_array(0)
		return author
	}

	def main(args: Array[String]) {
		val conf = new SparkConf().setAppName("Simple Application")
		val sc = new SparkContext(conf)
		val sqlContext = new SQLContext(sc)
		import sqlContext.implicits._
		//path :: the path of original books
		
		//D:\project\books\tmp\raw
		//val path = "D:/project/books/tmp/raw"
		//val file_split_path = "D:/project/books/tmp/split"
		val path = args(0)
		val file_split_path = args(1)
		//val input_df = sc.wholeTextFiles(path)
		//val name_df = input_df.toDF.map(s=>s(0).toString)
		//val bookname_iterator = name_df.toLocalIterator()
		val bookname_iterator = subdirs(new File(path))
		//w2v 模型加载地址："D:/tmp/w2c"
		//val w2c_model = Word2VecModel.load("D:/tmp/w2c")
		val w2c_model = Word2VecModel.load(args(2))
		//lr 模型加载地址："D:/tmp/model"
		//val mlr_model = LogisticRegressionModel.load("D:/tmp/model")
		val mlr_model = LogisticRegressionModel.load(args(3))
		
		//add
		val sub_obj_model = LogisticRegressionModel.load(args(4))
		//add
		
		//val lanscape_words_path = "D:/Learn/hku/project/signeddata/lanscape description words.txt"
		val lanscape_words_path = args(5)
		var lanscape_string = ""
		for(line<-Source.fromFile(lanscape_words_path,"utf-8").getLines())
			//for(line<-Source.fromBytes(bookname_path).getLines())
			lanscape_string = lanscape_string + line
		val lanscape_words_array = lanscape_string.split(" ")
		val lanscape_set = Set("")
		for(lanscape_word<-lanscape_words_array)
			lanscape_set.add(lanscape_word)
		lanscape_set.remove("")
		
		///add
		val knight_words_path = args(6)
		var knight_string = ""
		for(line<-Source.fromFile(knight_words_path,"utf-8").getLines())
			//for(line<-Source.fromBytes(bookname_path).getLines())
			knight_string = knight_string + line
		val knight_words_array = knight_string.split(" ")
		val knight_set = Set("")
		for(knight_word<-knight_words_array)
			knight_set.add(knight_word)
		knight_set.remove("")
		
		val middle_words_path = args(7)
		var middle_string = ""
		for(line<-Source.fromFile(middle_words_path,"utf-8").getLines())
			//for(line<-Source.fromBytes(bookname_path).getLines())
			middle_string = middle_string + line
		val middle_words_array = middle_string.split(" ")
		val middle_set = Set("")
		for(middle_word<-middle_words_array)
			middle_set.add(middle_word)
		middle_set.remove("")
		///add
		
		
		
		
		
		while(bookname_iterator.hasNext){
			val bookname_path = bookname_iterator.next()
			var file_string = ""
			for(line<-Source.fromFile(bookname_path,"utf-8").getLines())
			//for(line<-Source.fromBytes(bookname_path).getLines())
				file_string = file_string + "\n" + line
				//file_string = file_string + " " + line
			val file_length = file_string.length
			val file_start = file_string.substring(file_length/10,file_length/10*3)
			val file_mid = file_string.substring(file_length/10*4,file_length/10*6)
			val file_finish = file_string.substring(file_length/10*7,file_length/10*9)
			val file_temp = file_string.substring(1000,6000)
			val temp_array = file_temp.split(" ")
			var record = 0
			for(temp<-temp_array)
				if(lanscape_set.exists(_==temp))
					record = record + 1
			println("=====")
			println(record)
			println("=====")
			
			
			
			//add
			var knight_record = 0
			for(temp<-temp_array)
				if(knight_set.exists(_==temp))
					knight_record = knight_record + 1
			println("=====")
			println(knight_record)
			println("=====")
			var middle_record = 0
			for(temp<-temp_array)
				if(middle_set.exists(_==temp))
					middle_record = middle_record + 1
			println("=====")
			println(middle_record)
			println("=====")
			//add
			
			
			val bookname = return_bookname(bookname_path.toString)
			val author = return_author(bookname_path.toString)
			println(bookname)
			println(author)
			val file_start_split = file_start.replaceAll("\n"," ").split("[\\,,\\.,\\?,\\!,\\:,\\;]")
			val file_mid_split = file_mid.replaceAll("\n"," ").split("[\\,,\\.,\\?,\\!,\\:,\\;]")
			val file_finish_split = file_finish.replaceAll("\n"," ").split("[\\,,\\.,\\?,\\!,\\:,\\;]")
			val start_path = file_split_path + "/" + bookname +"_start.txt"
			val mid_path = file_split_path + "/" + bookname +"_mid.txt"
			val finish_path = file_split_path + "/" + bookname +"_finish.txt"
			val writer_start = new PrintWriter(new File(start_path))
			for(line_start<-file_start_split)
				if(line_start.length>30)
					writer_start.write(line_start.trim()+"\n")
			writer_start.close()
			val writer_mid = new PrintWriter(new File(mid_path))
			for(line_mid<-file_mid_split)
				if(line_mid.length>30)
					writer_mid.write(line_mid.trim()+"\n")
			writer_mid.close()
			val writer_finish = new PrintWriter(new File(finish_path))
			for(line_finish<-file_finish_split)
				if(line_finish.length>30)
					writer_finish.write(line_finish.trim()+"\n")
			writer_finish.close()
			
			val df_file_start = sc.textFile("file://"+start_path,2)
			val df_file_mid = sc.textFile("file://"+mid_path,2)
			val df_file_finish = sc.textFile("file://"+finish_path,2)
			
			//println(df_file.count)
			val df_file_vec_start = df_file_start.map(s=>s.split(" ")).toDF("text")
			val df_w2c_start = w2c_model.transform(df_file_vec_start)
			val lr_result_start = mlr_model.transform(df_w2c_start)
			val begin_emotion = lr_result_start.filter(s=>(s(4)==1)).count.toDouble/lr_result_start.count.toDouble
			
			val df_file_vec_mid = df_file_mid.map(s=>s.split(" ")).toDF("text")
			val df_w2c_mid = w2c_model.transform(df_file_vec_mid)
			val lr_result_mid = mlr_model.transform(df_w2c_mid)
			val mid_emotion = lr_result_mid.filter(s=>(s(4)==1)).count.toDouble/lr_result_mid.count.toDouble
			
			val df_file_vec_finish = df_file_finish.map(s=>s.split(" ")).toDF("text")
			val df_w2c_finish = w2c_model.transform(df_file_vec_finish)
			val lr_result_finish = mlr_model.transform(df_w2c_finish)
			val finish_emotion = lr_result_finish.filter(s=>(s(4)==1)).count.toDouble/lr_result_finish.count.toDouble
			
			println(finish_emotion)
			
			
			//add
			val sub_obj_start = sub_obj_model.transform(df_w2c_start)
			val begin_sub_obj = sub_obj_start.filter(s=>(s(4)==1)).count.toDouble/sub_obj_start.count.toDouble
			val sub_obj_mid = sub_obj_model.transform(df_w2c_mid)
			val mid_sub_obj = sub_obj_mid.filter(s=>(s(4)==1)).count.toDouble/sub_obj_mid.count.toDouble
			val sub_obj_finish = sub_obj_model.transform(df_w2c_finish)
			val finish_sub_obj = sub_obj_finish.filter(s=>(s(4)==1)).count.toDouble/sub_obj_finish.count.toDouble
			//add
			println("==========")
			println(begin_emotion)
			println("==========")
			println("==========")
			println(mid_emotion)
			println("==========")
			println("==========")
			println(finish_emotion)
			println("==========")
			println("==========")
			println(begin_sub_obj)
			println("==========")
			println("==========")
			println(mid_sub_obj)
			println("==========")
			println("==========")
			println(finish_sub_obj)
			println("==========")
			
			
			val book_name = bookname
			val pos_neg_all = "null"
			val pos_neg_beg = begin_emotion
			val pos_neg_mid = mid_emotion
			val pos_neg_end = finish_emotion
			/*
			val sql_detectexists = sqlContext.sql("SELECT * FROM book_tb1 where book_name = "+bookname)
			
			if(sql_detectexists.count==0){
				//val sql_insert = sqlContext.sql("Insert INTO table book_tb1 VALUES ('test','test',1,0,2,1,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL)")
				val sql_insert = sqlContext.sql("Insert INTO table book_tb1 VALUES ("+bookname+","+author+","+"NULL"+","+begin_emotion+","+mid_emotion+","+finish_emotion+",NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL)")

			}
			else{
				val df_sql = sql_detectexists.take(1)
				
				val sub_obj_all = df_sql(0)(6)
				val sub_obj_beg = df_sql(0)(7)
				val sub_obj_mid = df_sql(0)(8)
				val sub_obj_end = df_sql(0)(9)
				val lanscape_fre = df_sql(0)(10)
				val knight_fre = df_sql(0)(11)
				val midage_fre = df_sql(0)(12)
				val rich_of_dif = df_sql(0)(13)
				val len_of_sen = df_sql(0)(14)
				val len_of_words = df_sql(0)(15)
				val stopwords_fre = df_sql(0)(16)
				val rich_of_dif_stop = df_sql(0)(17)
				val rich_of_dif_unstop = df_sql(0)(18)
				val topic = df_sql(0)(19)
				val date = df_sql(20)
				val sql_insert = sqlContext.sql("Insert INTO table book_tb1 VALUES ("+book_name+","+author+","+"NULL"+","+pos_neg_all+","+pos_neg_mid+","+pos_neg_end+","+sub_obj_all+","+sub_obj_beg+","+sub_obj_mid+","+sub_obj_end+","+lanscape_fre+","+knight_fre+","+midage_fre+","+rich_of_dif+","+len_of_sen+","+len_of_words+","+stopwords_fre+","+rich_of_dif_stop+","+rich_of_dif_unstop+","+topic+","+date+")")
			}
			*/
			
		}
		
		
	}
	}