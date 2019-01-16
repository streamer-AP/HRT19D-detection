#include<iostream>
#include <fstream>
#include"Naive_Bayes.h"
using namespace std;

int main()
{ 
	Naive_Bayes a(58);//参数加上结果共58个
	a.train("train.data");//训练集
	a.test("test.data");//测试集
	a.predict("get_the_flag.data");
}