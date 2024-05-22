#ifndef __OCTREE_SERIALIZER__
#define __OCTREE_SERIALIZER__
#include "GaussianOctree.h"
#include "GaussianBVH.h"

class SerializedOctree{
private:
    uint64_t key;
    /*
        4 bits: Level in the octree, can represent up to 16 levels (0 to 15)
        48 bits: Position encoding, defines the position inside the parent, 3 bits per level
        1 bit: Is Leaf flag
        10 bits: Encodes the number of splats contained in the node
    
    */

public: 
    void setLevel(int level);
    int getLevel();

    void setEncoding(int position);
    int getEncoding();

    void setIsLeaf(int isLeaf);
    int getIsLeaf();

    void setNumSplats(int numSplats);
    int getNumSplats();
};

void SerializedOctree::setLevel(int level){
    uint64_t mask = (1 << 61) - 1;
    mask += (level << 60);
    key = key | mask;
}
int SerializedOctree::getLevel(){
    return key >> 60;
}

void SerializedOctree::setEncoding(int position){

}
int SerializedOctree::getEncoding(){

}

void SerializedOctree::setIsLeaf(int isLeaf){

}
int SerializedOctree::getIsLeaf(){

}

void SerializedOctree::setNumSplats(int numSplats){

}
int SerializedOctree::getNumSplats(){

}

#endif