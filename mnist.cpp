#include <cstring>
#include <string>
#include <unordered_set>
#include <memory>
#include <functional>
#include <vector>
#include <iostream>
#include "engine.h"
#include "nn.h"
#include <curl/curl.h>
#include <filesystem>
#include <fstream>
#include <chrono>

namespace fs = std::filesystem;
typedef unsigned char uchar;

static std::string train_imgs_url = "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz";
static std::string train_labels_url = "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz";
static std::string data_out_path = "data/train-ubyte.gz";
static std::string labels_out_path = "data/train-labels-ubyte.gz";
static std::string data_path = "data/train-ubyte";
static std::string label_path = "data/train-labels-ubyte";

inline bool file_exists(const std::string& name) {
    if (FILE *file = fopen(name.c_str(), "r")) {
        fclose(file);
        return true;
    } else {
        return false;
    }   
}

size_t write_data(void *ptr, size_t size, size_t nmemb, FILE *stream) {
    size_t written = fwrite(ptr, size, nmemb, stream);
    return written;
}

void mkdir(const std::string& path) {
    if(fs::create_directory(path)) {
        std::cout << "Created directory " << path << std::endl;
    } else {
        std::cout << "Failed to create directory " << path << std::endl;
    }
}

void download_and_unzip(const std::string& url, const std::string &out_path) {
    CURL *curl;
    FILE *fp;
    CURLcode res;
    curl = curl_easy_init();

    if(!file_exists("data")) {
        mkdir("data");
    }

    if (curl) {
        std::cout << "Downloading MNIST data from " << url << " into " << out_path << std::endl;
        fp = fopen(out_path.c_str(),"wb");
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_data);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, fp);
        res = curl_easy_perform(curl);

        if(res != CURLE_OK) {
            std::cout << "Failed to download " << out_path << std::endl;
        } else {
            std::cout << "Successfully downloaded into " << out_path << std::endl;
        }

        /* always cleanup */
        curl_easy_cleanup(curl);
        fclose(fp);

        std::cout << "Decompressing " << out_path << std::endl;
        std::string syscall = "gzip -d "+out_path;
        system(syscall.c_str());
    }
}

void make_data() {
    download_and_unzip(train_imgs_url, data_out_path);
    download_and_unzip(train_labels_url, labels_out_path);
}

//for use in reading data
int reverseInt (int i) 
{
    unsigned char c1, c2, c3, c4;

    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;

    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

//read all images into a vector then return pointer to vector of values
std::shared_ptr<std::vector<std::vector<int>>> read_mnist(std::string full_path)
{
    std::ifstream file(full_path, std::ios::binary);
    if (file.is_open())
    {
        std::cout << "Reading MNIST Data..." << std::endl;
        int magic_number=0;
        int number_of_images=0;
        int n_rows=0;
        int n_cols=0;
        file.read((char*)&magic_number,sizeof(magic_number)); 
        magic_number= reverseInt(magic_number);
        file.read((char*)&number_of_images,sizeof(number_of_images));
        number_of_images= reverseInt(number_of_images);
        file.read((char*)&n_rows,sizeof(n_rows));
        n_rows= reverseInt(n_rows);
        file.read((char*)&n_cols,sizeof(n_cols));
        n_cols= reverseInt(n_cols);

        //pointer to vector that holds all images
        auto all_images = std::make_shared<std::vector<std::vector<int>>>();
        for(int i=0;i<number_of_images;++i)
        {
            std::vector<int> img;
            img.reserve(n_rows*n_cols);
            for(int r=0;r<n_rows;++r)
            {
                for(int c=0;c<n_cols;++c)
                {
                    unsigned char temp=0;
                    file.read((char*)&temp,sizeof(temp));
                    img.push_back((int)temp);
                }
            }
            all_images -> push_back(img);
        }
        std::cout << "Done reading MNIST Data!" << std::endl;
        return all_images;
    }
    throw std::runtime_error("Invalid MNIST image file!");
}

//return pointer to vector of labels
std::shared_ptr<std::vector<int>> read_mnist_labels(std::string full_path, int& number_of_labels) {
    std::ifstream file(full_path, std::ios::binary);

    if(file.is_open()) {
        std::cout << "Reading MNIST Labels..." << std::endl;
        int magic_number = 0;
        file.read((char *)&magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);
        if(magic_number != 2049) throw std::runtime_error("Invalid MNIST label file!");

        file.read((char *)&number_of_labels, sizeof(number_of_labels)), number_of_labels = reverseInt(number_of_labels);

        auto labels = std::make_shared<std::vector<int>>(number_of_labels);
        for(int i = 0; i < number_of_labels; i++) {
            unsigned char temp = 0;
            file.read((char *)&temp, sizeof(temp));
            (*labels)[i] = (int)temp;
        }
        std::cout << "Done reading MNIST Labels!" << std::endl;
        return labels;
    } else {
        throw std::runtime_error("Unable to open file `" + full_path + "`!");
    }
}

std::vector<std::shared_ptr<Value>> convert_to_values(std::vector<int>& arr) {
    std::vector<std::shared_ptr<Value>> out;
    out.reserve(arr.size());
    for(auto& pixel: arr) {
        out.push_back(std::make_shared<Value>(pixel));
    }
    return out;
}

int main(int argc, char *argv[]) {

    bool cuda = false;

    if(argc > 1 && strcmp("--cuda", argv[1]) == 0) {
        cuda = true;
    }

    std::string path = "data/";
    
    if(!file_exists(path)) {
        make_data();
    }

    auto images = read_mnist(data_path);
    int size = images -> size();
    auto labels = read_mnist_labels(label_path, size);

    MLP model = MLP({784, 100, 100, 10}, "leaky_relu");
    auto epochs = 1;
    auto step = 0;
    auto lr = .05;

    std::cout << "Training..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    while(epochs--) {
        for(int i = 0; i < 40000; i++) {
            model.zero_grad();
            auto x = convert_to_values((*images)[i]);
            auto label = (*labels)[i];
            std::vector<std::shared_ptr<Value>> truth;
            for(int j = 0; j < 10; j++) {
                if(j == label) {
                    truth.push_back(std::make_shared<Value>(1));
                } else {
                    truth.push_back(std::make_shared<Value>(0));
                }
            }
            auto y = model(x);
            y = softmax(y);
            auto loss = cross_entropy(y, truth);
            loss->backward();
            model.step(lr);
            if(step%10==0) {
                std::cout << "Step: " << step << " Loss: " << loss-> get_data() << std::endl;
            }
            step++;
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end-start).count();
    std::cout << "Training took " << duration << " seconds" << std::endl;
    std::cout << "Done training!" << std::endl;
    return 0;
}
