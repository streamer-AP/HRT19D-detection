#include<iostream>
#include<string>
#include<math.h>
using namespace std;
#define PI 3.1415926
#define E 2.71828
#define Normal_distribution(x,u,a) (1/(sqrt(2*PI)*a))*pow(E,-1*(x-u)*(x-u)/(2*a*a))
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//类:朴素贝叶斯分类
class Naive_Bayes
{
public:
	Naive_Bayes(int feature_quantity_);
	~Naive_Bayes();
	void train(string path);
	void test(string path);
	void predict(string path);
private:
	int feature_quantity;
	double* Features;
	double* Features_mean_value_type_0;
	double* Features_mean_value_type_1;
	double* Features_variance_type_0;
	double* Features_variance_type_1;
	double p_type_1;
	double p_type_0;
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//构造函数
//参数:特征数量
Naive_Bayes::Naive_Bayes(int feature_quantity_)
{
	feature_quantity = feature_quantity_;
	Features = new double [feature_quantity];
	Features_mean_value_type_0 = new double [feature_quantity-1];
	Features_mean_value_type_1 = new double [feature_quantity-1];
	Features_variance_type_0 = new double [feature_quantity-1];
	Features_variance_type_1 = new double [feature_quantity-1];
	int i;
	for( i=0 ; i<feature_quantity-1 ; i++ )
	{
		Features_mean_value_type_0[i] = 0 ;
		Features_mean_value_type_1[i] = 0 ;
		Features_variance_type_0[i] = 0 ;
		Features_variance_type_1[i] = 0 ;
	}
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//析构函数
Naive_Bayes::~Naive_Bayes()
{
	delete [feature_quantity] Features;
	//delete [feature_quantity-1] Features_mean_value_type_0;
	//delete [feature_quantity-1] Features_mean_value_type_1;
	//delete [feature_quantity-1] Features_variance_type_0;
	//delete [feature_quantity-1] Features_variance_type_1;
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//函数:训练
//参数:所要读取文件的路径
void Naive_Bayes::train(string path)
{
	cout<<"calculating........"<<endl;
	int i;
	ifstream train_gathering( path.c_str() , ios::in );
	int test_quantity = 0 ;
	int type_1=0,type_0=0;
	if(train_gathering.is_open() != 1) 
	{
		cout<<"no file"<<endl;
	}
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//得到各个特征的均值
	while ( train_gathering.peek() != EOF )//读取特征
	{
		for( i=0 ; i<feature_quantity ; i++ )
		{
			train_gathering>>Features[i];
			train_gathering.seekg( 1 , ios::cur );
		}
		test_quantity++;
		if( Features[feature_quantity-1] == 1)
		{
			type_1++;
			for( i=0 ; i<feature_quantity ; i++ )
			{
				Features_mean_value_type_1[i] +=  Features[i];
			}
		}
		else if( Features[feature_quantity-1] == 0)
		{
			type_0++;
			for( i=0 ; i<feature_quantity ; i++ )
			{
				Features_mean_value_type_0[i] +=  Features[i];
			}
		}

	}
	for( i=0 ; i<feature_quantity ; i++ )//得到各个特征的均值
	{
		Features_mean_value_type_1[i] /= type_1;
		Features_mean_value_type_0[i] /= type_0;
	}
	////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//得到各个特征的标准差
	train_gathering.clear(std::ios::goodbit);//回到文件开头
	train_gathering.seekg( 0 , ios::beg );

	while ( train_gathering.peek() != EOF )//读取特征
	{
		for( i=0 ; i<feature_quantity ; i++ )
		{
			train_gathering>>Features[i];
			train_gathering.seekg( 1 , ios::cur );
		}
		if( Features[feature_quantity-1] == 1)
		{
			for( i=0 ; i<feature_quantity ; i++ )
			{
				Features_variance_type_1[i] += ( Features[i]-Features_mean_value_type_1[i] )*( Features[i]-Features_mean_value_type_1[i] );
			}
		}
		else if( Features[feature_quantity-1] == 0)
		{
			for( i=0 ; i<feature_quantity ; i++ )
			{
				Features_variance_type_0[i] += ( Features[i]-Features_mean_value_type_0[i] )*( Features[i]-Features_mean_value_type_0[i] );
			}
		}
	}
	for( i=0 ; i<feature_quantity-1 ; i++ )//得到各个特征的标准差
	{
		Features_variance_type_0[i] /= test_quantity;
		Features_variance_type_0[i] = sqrt(Features_variance_type_0[i]);
		Features_variance_type_1[i] /= test_quantity;
		Features_variance_type_1[i] = sqrt(Features_variance_type_1[i]);
	}
	train_gathering.close();
	p_type_0 = 1.0*type_0 / test_quantity;
	p_type_1 = 1.0*type_1 / test_quantity;

	cout<<"calculated."<<endl;
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//函数:测试
//参数:所要读取文件的路径
void Naive_Bayes::test(string path)
{
	int number=0,i;
	double p_1;
	double p_0;
	int num_right=0;
	ifstream test_gathering( path.c_str() , ios::in );
	if(test_gathering.is_open() != 1) 
	{
		cout<<"no file"<<endl;
	}
	while ( test_gathering.peek() != EOF )//读取特征
	{
		number++;
		p_1=p_type_1;
		p_0=p_type_0;
		for( i=0 ; i<feature_quantity ; i++ )
		{
			test_gathering>>Features[i];
			test_gathering.seekg( 1 , ios::cur );

			if( i != feature_quantity-1 )
			{
				p_1 *= Normal_distribution( Features[i] , Features_mean_value_type_1[i] , Features_variance_type_1[i] );
				p_0 *= Normal_distribution( Features[i] , Features_mean_value_type_0[i] , Features_variance_type_0[i] );
			}
		}
		cout<<"分类结果:"<< (p_1>p_0?1:0) <<"  实际:"<<Features[feature_quantity -1 ]<<endl;
		num_right += (p_1>p_0?1:0 == Features[feature_quantity -1 ])?1:0;
	}
	cout<<"测试用例共计"<<number<<"个,"<<"分类正确个数:"<<num_right<<"  正确率:"<<1.0*num_right/number<<endl;
	
	test_gathering.close();
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//函数:预测
//参数:所要读取文件的路径
void Naive_Bayes::predict(string path)
{
	int number=0,i,k=1,k_1=0;
	double p_1;
	double p_0;
	int num_right=0;
	ifstream test_gathering( path.c_str() , ios::in );
	if(test_gathering.is_open() != 1) 
	{
		cout<<"no file"<<endl;
	}
	while ( test_gathering.peek() != EOF )//读取特征
	{
		number++;
		p_1=p_type_1;
		p_0=p_type_0;
		for( i=0 ; i<feature_quantity-1 ; i++ )
		{
			test_gathering>>Features[i];
			test_gathering.seekg( 1 , ios::cur );
			p_1 *= Normal_distribution( Features[i] , Features_mean_value_type_1[i] , Features_variance_type_1[i] );
			p_0 *= Normal_distribution( Features[i] , Features_mean_value_type_0[i] , Features_variance_type_0[i] );
		}
		k++;
		k_1+=(p_1>p_0?1:0);
		if( k == 10 )
		{
			cout<< (k_1>5?1:0 );
			k_1=0;
			k=0;
		}
	}
	cout<<"预测用例共计"<<number<<"个"<<endl;
	
	test_gathering.close();
}