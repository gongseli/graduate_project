import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.spark.sql.SQLContext
import org.apache.spark.sql.SQLContext
import org.apache.spark.ml.feature.Word2Vec
import org.apache.spark.ml.feature.Word2VecModel
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.classification.LogisticRegressionModel
object SimpleApp {

	def main(args: Array[String]) {
		
		val conf = new SparkConf().setAppName("Simple Application")
		val sc = new SparkContext(conf)
		val sqlContext = new SQLContext(sc)
		import sqlContext.implicits._
		//训练Wordvec数据集
		//val path_wrod2vec = "D:/tmp/word2vec.txt"
		//val path_pos = "D:/tmp/result_pos.txt"
		//val path_neg = "D:/tmp/result_neg.txt"
		val path_wrod2vec = args(0)
		val path_pos = args(1)
		val path_neg = args(2)
		
		val path_sub = args(3)
		val path_obj = args(4)
		
		
		val rdd_pos = sc.textFile(path_pos,2)
		val rdd_neg = sc.textFile(path_neg,2)
		val rdd_pos_sign = rdd_pos.map(s=>(s,1))
		val rdd_neg_sign = rdd_neg.map(s=>(s,0))
		
		val rdd_sub = sc.textFile(path_sub,2)
		val rdd_obj = sc.textFile(path_obj,2)
		val rdd_sub_sign = rdd_sub.map(s=>(s,1))
		val rdd_obj_sign = rdd_obj.map(s=>(s,0))
		
		val rdd_wrod2vec = sc.textFile(path_wrod2vec,2)
		
		//val doc_word2vec = rdd_wrod2vec.map(s=>s.split(" ")).map(Tuple1.apply).toDF("text")
		//val doc_pos = rdd_pos.map(s=>s.split(" ")).map(Tuple1.apply).toDF("text")
		//val doc_neg = rdd_neg.map(s=>s.split(" ")).map(Tuple1.apply).toDF("text")
		val doc_word2vec = rdd_wrod2vec.map(s=>s.split(" ")).toDF("text")
		val doc_pos = rdd_pos_sign.map(s=>(s._1.toString.split(" "),s._2)).toDF("text","label")
		val doc_neg = rdd_neg_sign.map(s=>(s._1.toString.split(" "),s._2)).toDF("text","label")
		
		val doc_sub = rdd_sub_sign.map(s=>(s._1.toString.split(" "),s._2)).toDF("text","label")
		val doc_obj = rdd_obj_sign.map(s=>(s._1.toString.split(" "),s._2)).toDF("text","label")
		
		//val_doc_pos_sign = doc_pos.map(Tuple1.apply)
		
		//val documentDF = sqlContext.createDataFrame(Seq("Hi I heard about Spark".split(" "),"I wish Java could use case classes".split(" "),"Logistic regression models are neat".split(" ")).map(Tuple1.apply)).toDF("text")
		//val documentDF = sqlContext.createDataFrame(Seq("Hi I heard about Spark","I wish Java could use case classes","Logistic regression models are neat").map(Tuple1.apply)).toDF("text")
		val word2Vec = new Word2Vec().setInputCol("text").setOutputCol("features").setVectorSize(3).setMinCount(0)
		val model = word2Vec.fit(doc_word2vec)
		
		
		val pos_df = model.transform(doc_pos)
		val neg_df = model.transform(doc_neg)
		//val pos_df_sign = pos_df.map(s=>(s(0)))
		val pos_neg_df = pos_df.union(neg_df)
		val mlr = new LogisticRegression().setMaxIter(10)
		val mlr_model = mlr.fit(pos_neg_df)
		
		val sub_df = model.transform(doc_sub)
		val obj_df = model.transform(doc_obj)
		val sub_obj_df = sub_df.union(obj_df)
		val obj_sub_model = mlr.fit(sub_obj_df)
		
		
		
		val path_cross = args(5)
		val rdd_cross = sc.textFile(path_cross,2)
		val doc_cross = rdd_cross.map(s=>s.split(" ")).toDF("text")
		val cross_df = model.transform(doc_cross)
		val lr_result = mlr_model.transform(cross_df)
		
		
		
		val path_w2cmodel = args(6)
		val path_pos_neg_model = args(7)
		val path_sub_obj_model = args(8)
		model.save(path_w2cmodel)
		mlr_model.save(path_pos_neg_model)
		obj_sub_model.save(path_sub_obj_model)
	}
}