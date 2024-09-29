// Reader for splat files in binary PLY format

#pragma once
#include <fstream>
#include <string>
#include "raster_helper.cuh"

int loadSplatData(std::string path, std::vector<SplatData> & dataBuffer, int * numElements){

    int dataSize = 0;

    float shBuffer[48];

    std::ifstream is(path.c_str());
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

        std::getline(is, crt_line); // finish line

        for(int i=0;i<62;i++){
            // pass over the fields, consider them standard for now
            std::getline(is, crt_line);
        }

        std::getline(is, crt_line); // end_header

        // Now we read the binary data
        SplatDataRaw * sdr = new SplatDataRaw[dataSize];
        is.read((char*)sdr, sizeof(SplatDataRaw) * dataSize);

        /* Convert data to correct format: scale as exponential, opacity is sigmoid, precompute 3D covariances*/
        dataBuffer.reserve(dataSize);
        dataBuffer.resize(dataSize);

        for(int i = 0; i < dataSize; i++){
            /* Scale is an exponential */
            for(int comp = 0; comp < 3; comp++){
                sdr[i].fields.scale[comp] = exp(sdr[i].fields.scale[comp]);
            }

            /* Opacity is sigmoid */
            sdr[i].fields.opacity = 1.0f / (1.0f + exp(-sdr[i].fields.opacity));

            memcpy(shBuffer, sdr[i].fields.SH, 48 * sizeof(float));
            /* Reorder SH components to have consecutive RGB */
            for(int j=1;j<16;j++){
                sdr[i].fields.SH[j * 3 + 0] = shBuffer[(j-1) + 3];
                sdr[i].fields.SH[j * 3 + 1] = shBuffer[(j-1) + 16 + 2];
                sdr[i].fields.SH[j * 3 + 2] = shBuffer[(j-1) + 2 * 16 + 1];
            }

            /* compy everything except the stuff needed for Cov3D */
            memcpy(dataBuffer[i].rawData, sdr[i].rawData, sizeof(float) * (sizeof(SplatData) / sizeof(float) - 15));

            glm::vec3 e1(sdr[i].fields.scale[0], 0.0f, 0.0f);
            glm::vec3 e2(0.0f, sdr[i].fields.scale[1], 0.0f);
            glm::vec3 e3(0.0f, 0.0f, sdr[i].fields.scale[2]);

            computeVecRotationFromQuaternion(sdr[i].fields.rotation, e1);
            computeVecRotationFromQuaternion(sdr[i].fields.rotation, e2);
            computeVecRotationFromQuaternion(sdr[i].fields.rotation, e3);
            dataBuffer[i].fields.directions[0] = e1.x;
            dataBuffer[i].fields.directions[1] = e1.y;
            dataBuffer[i].fields.directions[2] = e1.z;
            dataBuffer[i].fields.directions[3] = e2.x;
            dataBuffer[i].fields.directions[4] = e2.y;
            dataBuffer[i].fields.directions[5] = e2.z;
            dataBuffer[i].fields.directions[6] = e3.x;
            dataBuffer[i].fields.directions[7] = e3.y;
            dataBuffer[i].fields.directions[8] = e3.z;

            glm::vec3 normal;
            computeCov3D(sdr[i].fields.scale, 1.0f, sdr[i].fields.rotation, dataBuffer[i].fields.covariance, normal);
            sdr[i].fields.normal[0] = normal.x;
            sdr[i].fields.normal[1] = normal.y;
            sdr[i].fields.normal[2] = normal.z;
        }

        *numElements = dataSize;
        delete [] sdr;
        return 0;
    }
    else{
        return 2;
    }
}

