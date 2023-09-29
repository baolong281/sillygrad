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

namespace fs = std::filesystem;

static std::string train_imgs_url = "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz";

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

void make_data() {
    CURL *curl;
    FILE *fp;
    CURLcode res;
    std::string out_path = "data/train.gz";
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
        std::cout << res << std::endl;
        /* always cleanup */
        curl_easy_cleanup(curl);
        fclose(fp);
    }
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

    return 0;
}
