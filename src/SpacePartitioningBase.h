class SpacePartitioningBase 
{
public:
    virtual ~SpacePartitioningBase() = default;
    virtual void * buildVHStructure(std::vector<SplatData> &sd, uint32_t num_primitives, volatile int *progress);
};