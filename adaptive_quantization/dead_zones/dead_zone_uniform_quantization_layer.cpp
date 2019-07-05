#include <cstdio>
#include <cstring>
#include <string>
#include <algorithm>
#include <vector>

using namespace std;

#define eps 1e-7

int num_samples = 50000;
int num_enum_quant = 200;

int num_layers;
vector<float> *original_data_per_layer;
int *num_data_per_layer;
vector<int> dead_zone_ratios;
vector<int> quant_levels;

string list_filenames;
string list_filesizes;

vector<string> all_filenames_data;

vector<float> load_binary_data_from_file(string filename , int num_data);
void read_all_data_from_file();
float binary_search(vector<float> x , int ratio);
vector<float> random_sampling(vector<float> x , int num_samples);
void dead_zone_uniform_quantization(vector<float> x , int dead_zone_ratio , int quant_level , float &lambda , float &delta);

string input_path;
string output_path;

int main(int argc, char** argv){
    for(int i = 0 ; i < argc ; i ++){
        printf("%d: %s\n" , i , argv[i]);
    }

    if(argc < 7){
        printf("The number of command line parameters is not enough.\n");
        exit(1);
    }

    num_layers = stoi(argv[1]);
    list_filenames = argv[2];
    list_filesizes = argv[3];

    input_path = argv[6];
    output_path = argv[7];

    read_all_data_from_file();

    FILE *file_dead_zone_ratios = fopen(argv[4] , "r");
    int ratio;

    while(fscanf(file_dead_zone_ratios , "%d" , &ratio) != EOF)
        dead_zone_ratios.push_back(ratio);
    fclose(file_dead_zone_ratios);

    FILE *file_quant_levels = fopen(argv[5] , "r");
    int quant_level;

    while(fscanf(file_quant_levels , "%d" , &quant_level) != EOF)
        quant_levels.push_back(quant_level);
    fclose(file_quant_levels);

    //for(int i = 0 ; i < all_filenames_data.size() ; i ++)
        //printf("filename %s\n" , all_filenames_data[i].c_str());
    //exit(1);

    //FILE *test_out = fopen("/home/wangzhe/Documents/exp/exp_2019_3/3_13_dead_zone_uniform_quantization/Pro_dead_zone_uniform_quantization/codebooks/test.txt" , "w");
    //fprintf(test_out , "1 2 3\n");
    //fclose(test_out);
    //exit(1);

    for(int i = 0 ; i < num_layers ; i ++){
        for(int j = 0 ; j < dead_zone_ratios.size() ; j ++){
            for(int k = 0  ; k < quant_levels.size() ; k ++){
                float lambda, delta;
                dead_zone_uniform_quantization(original_data_per_layer[i] , dead_zone_ratios[j] , quant_levels[k] , lambda , delta);
                char filename_codebook[1010];
                sprintf(filename_codebook , "%s/%s_%d_%d.cb" , output_path.c_str() , all_filenames_data[i].c_str() , dead_zone_ratios[j] , quant_levels[k]);

                //printf("output filename %s\n" , filename_codebook);
                FILE *out = fopen(filename_codebook , "w");
                fprintf(out , "%.12f\n%.12f\n" , lambda , delta);
                fclose(out);

                printf("finish quantization layer %d dead zone %d quant level %d\n" , i + 1 , dead_zone_ratios[j] , quant_levels[k]);
            }
        }
    }

    return 0;
}


vector<float> load_binary_data_from_file(string filename , int num_data){
    FILE *file = fopen(filename.c_str() , "rb");
    if(file == NULL){
        printf("file not found: %s.\n" , filename.c_str());
        exit(1);
    }

    float *data = (float *)malloc(sizeof(float) * num_data);
    int num = fread(data , sizeof(float) , num_data , file);

    vector<float> rst;
    for(int i = 0 ; i < num_data ; i ++)
        rst.push_back(data[i]);

    fclose(file);
    delete[] data;

    printf("finish reading data from %s\n." , filename.c_str());
    return rst;
}

void read_all_data_from_file(){
    original_data_per_layer = new vector<float>[num_layers];
    num_data_per_layer = new int[num_layers];

    FILE *file_0 = fopen(list_filenames.c_str() , "r");
    if(file_0 == NULL){
        printf("file not found: %s.\n" , list_filenames.c_str());
        exit(1);
    }

    FILE *file_1 = fopen(list_filesizes.c_str() , "r");
    if(file_1 == NULL){
        printf("file not found: %s.\n" , list_filesizes.c_str());
        exit(1);
    }

    for(int i = 0 ; i < num_layers ; i ++){
        fscanf(file_1 , "%d" , &num_data_per_layer[i]);

        char filename[1010];
        fscanf(file_0 , "%s" , filename);

        string abs_path = input_path + filename;

        printf("begin loading data %d: " , i);
        original_data_per_layer[i] = load_binary_data_from_file(abs_path.c_str() , num_data_per_layer[i]);

        all_filenames_data.push_back(filename);
    }

    fclose(file_1);
    fclose(file_0);

    return ;
}

float binary_search(vector<float> x , int ratio){
	int total_number = x.size();

	float left = 0;
	float right = 0;

	for(int i = 0 ; i < total_number ; i ++){
		if(x[i] < 0){
			if(x[i] * -1.0 > right)
				right = x[i] * -1.0;
		}else{
			if(x[i] > right)
				right = x[i];
		}
	}

	right = right * 2.0;

	while(right - left > eps){
		float mid = (left + right) / 2.0;
		int cnt = 0;
		for(int i = 0 ; i < total_number ; i ++)
			if(std::abs(x[i]) <= mid)
				cnt ++;

		float percentage = 1.0 * cnt / total_number;

		if(percentage < 1.0 * ratio / 100){
			left = mid;
		}else{
			right = mid;
		}
	}

	return left;
}

vector<float> random_sampling(vector<float> x , int num_samples){
    if(num_samples >= x.size())
        return x;

    vector<float> y;

    int select = num_samples;
    int remaining = x.size();

    srand (time(NULL));

    for(int i = 0 ; i < x.size() ; i ++){
        if((rand() % remaining) < select){
            y.push_back(x[i]);
            select --;
        }
        remaining --;
    }

    return y;
}

void dead_zone_uniform_quantization(vector<float> x , int dead_zone_ratio , int quant_levels , float &lambda , float &delta){
    lambda = binary_search(x , dead_zone_ratio);

    vector<float> samples = random_sampling(x , num_samples);

    float bound = -1.0;

    for(int i = 0 ; i < x.size() ; i ++){
        if(x[i] < 0.0){
            if(-1.0 * x[i] > bound)
                bound = -1.0 * x[i];
        }else{
            if(x[i] > bound)
                bound = x[i];
        }
    }

    bound = bound * 1.5;

    float quant_start = lambda;
    float quant_end;
    float mse = 100000000;
    int num_quant_levels_one_side = (quant_levels - 1) / 2;

    for(int i = 1 ; i <= num_enum_quant ; i ++){
        quant_end = lambda + (bound - lambda) / (1.0 * num_enum_quant) * i;
        float test_delta = (quant_end - quant_start) / (1.0 * num_quant_levels_one_side);

        float test_mse = 0.0;
        for(int j = 0 ; j < samples.size() ; j ++){
            float value = samples[j];
            float quant_value;
            if(std::abs(value) < lambda){
                quant_value = 0.0;
            }else{
                float quant_sign = 1.0;
                float abs_value = value;
                if(value < 0.0){
                    quant_sign = -1.0;
                    abs_value = -1.0 * value;
                }

                int id = (int)((abs_value - lambda) / test_delta);
                if(id >= num_quant_levels_one_side){
                    id = num_quant_levels_one_side - 1;
                }

                quant_value = quant_sign * (lambda + test_delta / 2.0 + test_delta * id);
            }
            test_mse += (quant_value - value) * (quant_value - value);
        }

        if(test_mse < mse){
            mse = test_mse;
            delta = test_delta;
        }
    }

    return ;
}
