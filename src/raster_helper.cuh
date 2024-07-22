#pragma once
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <Eigen/Dense>

#define BLOCK_X 16
#define BLOCK_Y 16
#define BLOCK_SIZE (BLOCK_X * BLOCK_Y)

#define LINE_BLOCK 256

union SplatDataRaw
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

union SplatData
{
    float rawData[70]; // For faster reading, then we can split it into fields
    struct Fields
    {
        float position[3];
        float normal[3];
        float SH[48];
        float opacity;
		float directions[9];
        float covariance[6];
		
    } fields;
};

struct Frustum{
	float planes[6][4]; // 6 planes, each plane defined by 4 floats [(a, b, c), d]
};

__host__ void computeFrustum(Frustum & f,  glm::mat4 & mvp){
	// Left plane
    f.planes[0][0] = mvp[0][3] + mvp[0][0];
	f.planes[0][1] = mvp[1][3] + mvp[1][0];
	f.planes[0][2] = mvp[2][3] + mvp[2][0];
	f.planes[0][3] = mvp[3][3] + mvp[3][0];

    // Right plane
    f.planes[1][0] = mvp[0][3] - mvp[0][0];
	f.planes[1][1] = mvp[1][3] - mvp[1][0];
	f.planes[1][2] = mvp[2][3] - mvp[2][0];
	f.planes[1][3] = mvp[3][3] - mvp[3][0];

    // Bottom plane
	f.planes[2][0] = mvp[0][3] + mvp[0][1];
	f.planes[2][1] = mvp[1][3] + mvp[1][1];
	f.planes[2][2] = mvp[2][3] + mvp[2][1];
	f.planes[2][3] = mvp[3][3] + mvp[3][1];

    // Top plane
    f.planes[3][0] = mvp[0][3] - mvp[0][1];
	f.planes[3][1] = mvp[1][3] - mvp[1][1];
	f.planes[3][2] = mvp[2][3] - mvp[2][1];
	f.planes[3][3] = mvp[3][3] - mvp[3][1];

    // Near plane
	f.planes[4][0] = mvp[0][3] + mvp[0][2];
	f.planes[4][1] = mvp[1][3] + mvp[1][2];
	f.planes[4][2] = mvp[2][3] + mvp[2][2];
	f.planes[4][3] = mvp[3][3] + mvp[3][2];

    // Far plane
    f.planes[5][0] = mvp[0][3] - mvp[0][2];
	f.planes[5][1] = mvp[1][3] - mvp[1][2];
	f.planes[5][2] = mvp[2][3] - mvp[2][2];
	f.planes[5][3] = mvp[3][3] - mvp[3][2];
}

__device__ bool isPointInFrustum(const Frustum & f, const glm::vec3 & point, float radius){
	for(int i = 0; i < 6; i++){
		if(glm::dot(glm::make_vec3(f.planes[i]), point) + f.planes[i][3] < -radius){
			return false;
		}
	}
	return true;
}

// Spherical harmonics coefficients
__device__ const float SH_C0 = 0.28209479177387814f;
__device__ const float SH_C1 = 0.4886025119029199f;
__device__ const float SH_C2[] = {
	1.0925484305920792f,
	-1.0925484305920792f,
	0.31539156525252005f,
	-1.0925484305920792f,
	0.5462742152960396f
};
__device__ const float SH_C3[] = {
	-0.5900435899266435f,
	2.890611442640554f,
	-0.4570457994644658f,
	0.3731763325901154f,
	-0.4570457994644658f,
	1.445305721320277f,
	-0.5900435899266435f
};

__device__ float clip(float in, float min_val, float max_val){
    return min(max_val, max(in, min_val));
}

__host__ void printMat(glm::mat4 m){
	for(int i=0;i<4;i++){
		for(int j=0;j<4;j++){
			printf("%f ", m[i][j]);
		}
		printf("\n");
	}
	printf("-----------------------------\n");
}
__device__ void printMat(glm::mat3 m){
	for(int i=0;i<3;i++){
		for(int j=0;j<3;j++){
			printf("%f ", m[i][j]);
		}
		printf("\n");
	}
	printf("-----------------------------\n");
}

__device__ void insertCircularBuffer(uint32_t * data, size_t & begin, size_t & end, size_t bufferSize, uint32_t value){
	data[end] = value;
	end++;
	end = end % bufferSize;
}

__device__ uint32_t popCircularBuffer(uint32_t * data, size_t & begin, size_t & end, size_t bufferSize){
	uint32_t retVal = data[begin];
	begin++;
	begin = begin % bufferSize;
	return retVal;
}


__device__ glm::vec3 computeColorFromSH(int idx, int deg, const SplatData::Fields & sdFields, glm::vec3 & campos)
{
	// The implementation is loosely based on code for 
	// "Differentiable Point-Based Radiance Fields for 
	// Efficient View Synthesis" by Zhang et al. (2022)
	glm::vec3 pos = glm::make_vec3(sdFields.position);
	glm::vec3 dir = pos - campos;
	dir = dir / glm::length(dir);

	glm::vec3 * sh = (glm::vec3*) sdFields.SH;

	glm::vec3 result = SH_C0 * sh[0];

	if (deg > 0)
	{
		float x = dir.x;
		float y = dir.y;
		float z = dir.z;
		result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;
			result = result +
				SH_C2[0] * xy * sh[4] +
				SH_C2[1] * yz * sh[5] +
				SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
				SH_C2[3] * xz * sh[7] +
				SH_C2[4] * (xx - yy) * sh[8];

			if (deg > 2)
			{
				result = result +
					SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
					SH_C3[1] * xy * z * sh[10] +
					SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
					SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
					SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
					SH_C3[5] * z * (xx - yy) * sh[14] +
					SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
			}
		}
	}
	result += 0.5f;
	return glm::max(result, 0.0f);
}

void computeVecRotationFromQuaternion(const float * rot, glm::vec3 & vec){
	glm::vec4 q = glm::vec4(rot[0], rot[1], rot[2], rot[3]);
    q = q * (1.0f / glm::length(q));
	float q0 = q.x;
	float q1 = q.y;
	float q2 = q.z;
	float q3 = q.w;

	// Compute rotation matrix from quaternion: from https://automaticaddison.com/how-to-convert-a-quaternion-to-a-rotation-matrix/
	glm::mat3 R = glm::mat3(
		2.f * (q0 * q0 + q1 * q1) - 1.0f, 2.f * (q1 * q2 - q0 * q3), 2.f * (q1 * q3 + q0 * q2),
		2.f * (q1 * q2 + q0 * q3), 2.f * (q0 * q0 + q2 * q2) - 1.0f, 2.f * (q2 * q3 - q0 * q1),
		2.f * (q1 * q3 - q0 * q2), 2.f * (q2 * q3 + q0 * q1), 2.f * (q0 * q0 + q3 * q3) - 1.0f
	);

	vec = vec * R;

}

__forceinline__ __device__ float3 transformPoint4x3(const float3& p, const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
	};
	return transformed;
}

__device__ float3 computeCov2D(const glm::vec4& mean, float focal_x, float focal_y, float tan_fovx, float tan_fovy, const float* cov3D, const glm::mat4 view, float debug = false)
{
	// The following models the steps outlined by equations 29
	// and 31 in "EWA Splatting" (Zwicker et al., 2002). 
	// Additionally considers aspect / scaling of viewport.
	// Transposes used to account for row-/column-major conventions.
	glm::vec4 t = view * mean;

	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;

	glm::mat3 J = glm::mat3(
		focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
		0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
		0, 0, 0);

	glm::mat3 W = glm::mat3(
		view[0][0], view[1][0], view[2][0],
		view[0][1], view[1][1], view[2][1],
		view[0][2], view[1][2], view[2][2]);

	// W = glm::transpose(W);

	glm::mat3 T = W * J;

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	glm::mat3 cov = glm::transpose(T) * glm::transpose(Vrk) * T;
	// Apply low-pass filter: every Gaussian should be at least
	// one pixel wide/high. Discard 3rd row and column.
	cov[0][0] += 0.3f;
	cov[1][1] += 0.3f;
	return { float(cov[0][0]), float(cov[0][1]), float(cov[1][1]) };
}

/* __device__ if we want to use it again on the GPU */
void computeCov3D(const float * scale, float mod, const float * rot, float* cov3D, glm::vec3 & normal)
{
	// Create scaling matrix
	glm::mat3 S = glm::mat3(1.0f);
	S[0][0] = mod * scale[0];
	S[1][1] = mod * scale[1];
	S[2][2] = mod * scale[2];

    glm::vec4 q = glm::vec4(rot[0], rot[1], rot[2], rot[3]);
    q = q * (1.0f / glm::length(q));
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	// Compute rotation matrix from quaternion
	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	// Compute the normal orientation
	if(scale[0] < scale[1] && scale[0] < scale[2]){
		normal = glm::vec3(1.0f, 0.0f, 0.0f);
	}
	else if(scale[1] < scale[0] && scale[1] < scale[2]){
		normal = glm::vec3(0.0f, 1.0f, 0.0f);
	}
	else{
		normal = glm::vec3(0.0f, 0.0f, 1.0f);
	}

	normal = R * normal;

	glm::mat3 M = S * R;

	// Compute 3D world covariance matrix Sigma
	glm::mat3 Sigma = glm::transpose(M) * M;

	// Covariance is symmetric, only store upper right
	cov3D[0] = Sigma[0][0];
	cov3D[1] = Sigma[0][1];
	cov3D[2] = Sigma[0][2];
	cov3D[3] = Sigma[1][1];
	cov3D[4] = Sigma[1][2];
	cov3D[5] = Sigma[2][2];
}

__forceinline__ __device__ void getRect(const float2 p, int max_radius, uint2& rect_min, uint2& rect_max, dim3 grid)
{
	rect_min = {
		min(grid.x, max((int)0, (int)((p.x - max_radius) / BLOCK_X))),
		min(grid.y, max((int)0, (int)((p.y - max_radius) / BLOCK_Y)))
	};
	rect_max = {
		min(grid.x, max((int)0, (int)((p.x + max_radius + BLOCK_X - 1) / BLOCK_X))),
		min(grid.y, max((int)0, (int)((p.y + max_radius + BLOCK_Y - 1) / BLOCK_Y)))
	};
}
