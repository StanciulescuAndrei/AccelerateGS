#define OPACITY_THRESHOLD 0.2f

class SpacePartitioningBase 
{
public:
    virtual ~SpacePartitioningBase() = default;
    virtual void buildVHStructure(std::vector<SplatData> &sd, uint32_t num_primitives, volatile int *progress) = 0;
    virtual int markForRender(bool *renderMask, uint32_t num_primitives, int renderLevel, glm::vec3 &cameraPosition, float fovy, int SW, float dpt) = 0;
};