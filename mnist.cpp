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

namespace fs = std::filesystem;

static std::string train_imgs_url = "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz";
static std::string trai_labels_url = "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz";

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
        std::cout << "Downloading MNIST data from " << train_imgs_url << std::endl;
        fp = fopen(out_path.c_str(),"wb");
        curl_easy_setopt(curl, CURLOPT_URL, train_imgs_url.c_str());
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_data);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, fp);
        res = curl_easy_perform(curl);

        if(res != CURLE_OK) {
            std::cout << "Failed to download MNIST data" << std::endl;
        } else {
            std::cout << "Successfully downloaded MNIST data" << std::endl;
        }

        /* always cleanup */
        curl_easy_cleanup(curl);
        fclose(fp);

        std::cout << "Decompressing MNIST data" << std::endl;
        std::string syscall = "gzip -d "+out_path;
        system(syscall.c_str());
    }
}

void make_data() {
    std::string data_out_path = "data/train.gz";
    std:: string labels_out_path = "data/train-labels.gz";
    download_and_unzip(train_imgs_url, data_out_path);
    download_and_unzip(train_imgs_url, labels_out_path);
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

//read all images into a vector then return pointer to vector
std::shared_ptr<std::vector<std::vector<int>>> read_mnist(std::string full_path)
{
    std::ifstream file(full_path, std::ios::binary);
    if (file.is_open())
    {
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
        auto all_images = make_shared<std::vector<std::vector<int>>>(number_of_images, std::vector<int>(n_rows*n_cols));
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
            (*all_images)[i] = img;
        }
        return all_images;
    }
    return nullptr;
}

int main(int argc, char *argv[]) {

    bool cuda = false;

    if(argc > 1 && strcmp("-cuda", argv[1]) == 0) {
        cuda = true;
    }

    std::string path = "data/";
    
    if(!file_exists(path)) {
        make_data();
    }

    auto images = read_mnist("data/train");
    auto im1 = (*images)[0];

    for(int i = 0; i < 28; i++) {
        for(int j = 0; j < 28; j++) {
            std::cout << im1[i*28+j] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << images->size() << std::endl;

    return 0;
}
