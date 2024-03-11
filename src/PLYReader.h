// Reader for splat files in binary PLY format

#pragma once
#include <fstream>
#include <string>

union SplatData
{
    float rawData[62]; // For faster reading, then we can split it into fields
    struct Fields
    {
        float position[3];
        float normal[3];
        float SH[48];
        float opacity;
        float scale[3];
        float rotation[4];
    } fields;
};

int loadSplatData(char* path, SplatData ** dataBuffer, int * numElements){
    if(path == nullptr)
    {
        // invalid path
        return 1;
    }

    int dataSize = 0;

    float shBuffer[48];

    std::ifstream is(path);
    if(is.is_open()){
        std::string crt_line;
        std::getline(is, crt_line);
        printf("> %s\n", crt_line.c_str());

        if(crt_line.compare(std::string("ply"))!=0)
        {
            // not a ply
            return 3;
        }

        std::getline(is, crt_line); // format line

        is >> crt_line; // element
        is >> crt_line; // vertex
        is >> dataSize; // number of elements

        printf("Number of splats: %d\n", dataSize);
        std::getline(is, crt_line); // finish line

        for(int i=0;i<62;i++){
            // pass over the fields, consider them standard for now
            std::getline(is, crt_line);
        }

        std::getline(is, crt_line); // end_header

        // Now we read the binary data
        *dataBuffer = (SplatData*)malloc(sizeof(SplatData) * dataSize);
        is.read((char*)*dataBuffer, sizeof(SplatData) * dataSize);

        /* Convert data to correct format: scale as exponential, opacity is sigmoid */

        for(int i = 0; i < dataSize; i++){
            for(int comp = 0; comp < 3; comp++){
                (*dataBuffer)[i].fields.scale[comp] = exp((*dataBuffer)[i].fields.scale[comp]);
            }
            (*dataBuffer)[i].fields.opacity = 1.0f / (1.0f + exp(-(*dataBuffer)[i].fields.opacity));
            memcpy(shBuffer, (*dataBuffer)[i].fields.SH, 48 * sizeof(float));
            for(int j=1;j<16;j++){
                (*dataBuffer)[i].fields.SH[j * 3 + 0] = shBuffer[(j-1) + 3];
                (*dataBuffer)[i].fields.SH[j * 3 + 1] = shBuffer[(j-1) + 16 + 2];
                (*dataBuffer)[i].fields.SH[j * 3 + 2] = shBuffer[(j-1) + 2 * 16 + 1];
            }
            
        }

        *numElements = dataSize;

        return 0;
    }
    else{
        return 2;
    }
}

