#include <cstring>
#include <string>
#include <unordered_set>
#include <memory>
#include <functional>
#include <vector>
#include <iostream>
#include "engine.h"
#include "nn.h"

int main(int argc, char *argv[]) {

    bool cuda = false;

    if(argc > 1 && strcmp("-cuda", argv[1]) == 0) {
        cuda = true;
    }


    return 0;
}
