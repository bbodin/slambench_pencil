//#include "pencil.h"
//#include "vector_types.h"
//
#include <math.h>

#define min(X,Y) ((X) < (Y) ? (X) : (Y))
#define max(X,Y) ((X) > (Y) ? (X) : (Y))

typedef unsigned int uint;
typedef unsigned short ushort;

/* OpenCL vector type definitions for PENCIL. */
struct float3 {
	float x;
	float y;
	float z;
	float dummy; //for alignment as defined in Opencl standard
};

struct float4 {
	float x;
	float y;
	float z;
	float w;
};

struct uchar3 {
	unsigned char x;
	unsigned char y;
	unsigned char z;
};


struct uchar4 {
	unsigned char x;
	unsigned char y;
	unsigned char z;
	unsigned char w;
};

struct short2 {
	short x;
	short y;
};

struct float2 {
	float x;
	float y;
};

struct uint2 {
	unsigned int x;
	unsigned int y;
};

struct Matrix4 {
	 struct float4 data[4];
};

struct TrackData {
	int result;
	float error;
	float J[6];
};

typedef struct float3 float3;
typedef struct float4 float4;
typedef struct uchar3 uchar3;
typedef struct uchar4 uchar4;
typedef struct short2 short2;
typedef struct float2 float2;
typedef struct uint2 uint2;
typedef struct Matrix4 Matrix4;
typedef struct TrackData TrackData;

inline float3 make_float3(float x, float y, float z) {
	float3 ret;
	ret.x = x;
	ret.y = y;
	ret.z = z;
	return ret;
}

inline float4 make_float4(float x, float y, float z,float w) {
	float4 ret;
	ret.x = x;
	ret.y = y;
	ret.z = z;
	ret.w = w;
	return ret;
}

static float3 c_rotate(const Matrix4 M, const float3 v)
{
	return make_float3(M.data[0].x * v.x + M.data[0].y * v.y + M.data[0].z * v.z,
			M.data[1].x * v.x + M.data[1].y * v.y + M.data[1].z * v.z,
			M.data[2].x * v.x + M.data[2].y * v.y + M.data[2].z * v.z);
}


static void bilateralFilter_core_summary(int x, int y, int size_x, int size_y,
		int r, int gaussianS, float e_d,
		const float in[restrict const static size_y][size_x],
		const float gaussian[restrict const static gaussianS])
{
	__pencil_use(in[y][x]);
	for (int i = 0; i <= gaussianS; ++i) {
		__pencil_use(gaussian[i + r]);
	}
}

float bilateralFilter_core(int x, int y, int size_x, int size_y,
		int r, int gaussianS, float e_d,
		const float in[restrict const static size_y][size_x],
		const float gaussian[restrict const static gaussianS])
__attribute__((pencil_access(bilateralFilter_core_summary)));

static void integrateKernel_core_summary(const unsigned int vol_size_x, const unsigned int vol_size_y,
		const unsigned int vol_size_z, const float3 vol_dim,
		short2 vol_data[restrict const static vol_size_z][vol_size_y][vol_size_x],
		const unsigned int x, const unsigned int y, unsigned int depthSize_x, unsigned int depthSize_y,
		const float depth[restrict const static depthSize_y][depthSize_x],
		const Matrix4 invTrack, const Matrix4 K,
		const float mu, const float maxweight,
		const float3 delta, const float3 cameraDelta)
{
//	__pencil_use(delta[0]);
//	__pencil_use(cameraDelta[0]);
//	__pencil_use(vol_dim[0]);
	for (int z = 0; z <= vol_size_z; ++z) {
		__pencil_use(vol_data[z][y][x]);
		__pencil_def(vol_data[z][y][x]);
	}
	for (int i = 0; i < depthSize_y; i++)
	{
		for (int j = 0; j < depthSize_x; ++j)
		{
			 __pencil_use(depth[i][j]);
		}
	}
}

void integrateKernel_core(const unsigned int vol_size_x, const unsigned int vol_size_y,
		const unsigned int vol_size_z, const float3 vol_dim,
		short2 vol_data[restrict const static vol_size_z][vol_size_y][vol_size_x],
		const unsigned int x, const unsigned int y, unsigned int depthSize_x, unsigned int depthSize_y,
		const float depth[restrict const static depthSize_y][depthSize_x],
		const Matrix4 invTrack, const Matrix4 K,
		const float mu, const float maxweight,
		const float3 delta, const float3 cameraDelta)
__attribute__((pencil_access(integrateKernel_core_summary)));

static void initVolume_core_summary(const unsigned int x, const unsigned int y, const unsigned int z,
		const unsigned int v_size_x, const unsigned int v_size_y, const unsigned int v_size_z,
		short2 v_data[restrict const static v_size_z][v_size_y][v_size_x],
		const float dxVal, const float dyVal)
{
	__pencil_def(v_data[z][y][x]);
}

void initVolume_core(const unsigned int x, const unsigned int y, const unsigned int z,
		const unsigned int v_size_x, const unsigned int v_size_y, const unsigned int v_size_z,
		short2 v_data[restrict const static v_size_z][v_size_y][v_size_x],
		const float dxVal, const float dyVal)
__attribute__((pencil_access(initVolume_core_summary)));

static void depth2vertex_core_summary(unsigned int x, unsigned int y, unsigned int imageSize_x, unsigned int imageSize_y,
		const float depth[restrict const static imageSize_y][imageSize_x],
		const Matrix4 invK)
{
	const float depth_val = depth[y][x];
}

float3 depth2vertex_core(const unsigned int x, const unsigned int y,
		const unsigned int imageSize_x, const unsigned int imageSize_y,
		const float depth[restrict const static imageSize_y][imageSize_x],
		const Matrix4 invK)
__attribute__((pencil_access(depth2vertex_core_summary)));

static void vertex2normal_core_summary(const unsigned int x, const unsigned int y,
		const unsigned int imageSize_x, const unsigned int imageSize_y,
		const float3 in[restrict const static imageSize_y][imageSize_x])
{
	const float3 left = in[y][x];
}
float3 vertex2normal_core(const unsigned int x, const unsigned int y,
		const unsigned int imageSize_x, const unsigned int imageSize_y,
		const float3 in[restrict const static imageSize_y][imageSize_x])
__attribute__((pencil_access(vertex2normal_core_summary)));

static void halfSampleRobustImage_core_summary(const unsigned int x, const unsigned int y,
		const unsigned int outSize_x, const unsigned int outSize_y,
		const unsigned int inSize_x, const unsigned int inSize_y,
		const float in[restrict const static inSize_y][inSize_x],
		const float e_d, const int r)
{
	__pencil_use(in[y][x]);
}

float halfSampleRobustImage_core(const unsigned int x, const unsigned int y,
		const unsigned int outSize_x, const unsigned int outSize_y,
		const unsigned int inSize_x, const unsigned int inSize_y,
		const float in[restrict const static inSize_y][inSize_x],
		const float e_d, const int r)
__attribute__((pencil_access(halfSampleRobustImage_core_summary)));

static void renderNormal_core_summary(const unsigned int x, const unsigned int y,
		const unsigned int normalSize_x, const unsigned int normalSize_y,
		const float3 normal[restrict const static normalSize_y][normalSize_x])
{
	const float3 n = normal[y][x];
}

uchar3 renderNormal_core(const unsigned int x, const unsigned int y,
		const unsigned int normalSize_x, const unsigned int normalSize_y,
		const float3 normal[restrict const static normalSize_y][normalSize_x])
__attribute__((pencil_access(renderNormal_core_summary)));

static void renderDepth_core_summary(const unsigned int x, const unsigned int y,
		const unsigned int depthSize_x, const unsigned int depthSize_y,
		const float depth[restrict const static depthSize_y][depthSize_x],
		const float nearPlane, const float farPlane,
		const float rangeScale)
{
	const float d = depth[y][x];
}

uchar4 renderDepth_core(const unsigned int x, const unsigned int y,
		const unsigned int depthSize_x, const unsigned int depthSize_y,
		const float depth[restrict const static depthSize_y][depthSize_x],
		const float nearPlane, const float farPlane,
		const float rangeScale)
__attribute__((pencil_access(renderDepth_core_summary)));

static void renderTrack_core_summary(const unsigned int x, const unsigned int y,
		const unsigned int outSize_x, const unsigned int outSize_y,
		const TrackData data[restrict const static outSize_y][outSize_x])
{
	int test = data[y][x].result;
}

uchar4 renderTrack_core(const unsigned int x, const unsigned int y,
		const unsigned int outSize_x, const unsigned int outSize_y,
		const TrackData data[restrict const static outSize_y][outSize_x])
__attribute__((pencil_access(renderTrack_core_summary)));

static void renderVolume_core_summary(const unsigned int x, const unsigned int y,
		const unsigned int volume_size_x, const unsigned int volume_size_y, const unsigned int volume_size_z,
		const short2 volume_data[restrict const static volume_size_z][volume_size_y][volume_size_x],
		const float3 volume_dim, const Matrix4 view,
		const float nearPlane, const float farPlane,
		const float step, const float largestep,
		const float3 light, const float3 ambient)
{
	const short2 d = volume_data[x+y][y][x];
//	__pencil_use(volume_dim[0]);
//	__pencil_use(light[0]);
//	__pencil_use(ambient[0]);
}

uchar4 renderVolume_core(const unsigned int x, const unsigned int y,
		const unsigned int volume_size_x, const unsigned int volume_size_y, const unsigned int volume_size_z,
		const short2 volume_data[restrict const static volume_size_z][volume_size_y][volume_size_x],
		const float3 volume_dim, const Matrix4 view,
		const float nearPlane, const float farPlane,
		const float step, const float largestep,
		const float3 light, const float3 ambient)
__attribute__((pencil_access(renderVolume_core_summary)));

static void raycast_core_summary(const unsigned int x, const unsigned int y,
		const unsigned int inputSize_x, const unsigned int inputSize_y,
		float3 vertex[restrict const static inputSize_y][inputSize_x],
		float3 normal[restrict const static inputSize_y][inputSize_x],
		const unsigned int integration_size_x, const unsigned int integration_size_y, const unsigned int integration_size_z,
		const short2 integration_data[restrict const static integration_size_z][integration_size_y][integration_size_x],
		const float3 integration_dim, const Matrix4 view,
		const float nearPlane, const float farPlane,
		const float step, const float largestep)
{
	__pencil_use(integration_data[x+y][y][x]);
	__pencil_def(vertex[y][x]);
	__pencil_def(normal[y][x]);
}

void raycast_core(const unsigned int x, const unsigned int y,
		const unsigned int inputSize_x, const unsigned int inputSize_y,
		float3 vertex[restrict const static inputSize_y][inputSize_x],
		float3 normal[restrict const static inputSize_y][inputSize_x],
		const unsigned int integration_size_x, const unsigned int integration_size_y, const unsigned int integration_size_z,
		const short2 integration_data[restrict const static integration_size_z][integration_size_y][integration_size_x],
		const float3 integration_dim, const Matrix4 view,
		const float nearPlane, const float farPlane,
		const float step, const float largestep)
__attribute__((pencil_access(raycast_core_summary)));

static void track_core_summary(unsigned int refSize_x, unsigned int refSize_y, const TrackData output,
		const float3 inVertex, const float3 inNormal,
		const float3 refVertex[restrict const static refSize_y][refSize_x],
		const float3 refNormal[restrict const static refSize_y][refSize_x],
		const Matrix4 Ttrack, const Matrix4 view,
		const float dist_threshold,
		const float normal_threshold)
{
	const unsigned int refx;
	const unsigned int refy;
	const float3 vertVal = refVertex[refy][refx];
	const float3 normVal = refNormal[refy][refx];
}

TrackData track_core(unsigned int refSize_x, unsigned int refSize_y, const TrackData output,
		const float3 inVertex, const float3 inNormal,
		const float3 refVertex[restrict const static refSize_y][refSize_x],
		const float3 refNormal[restrict const static refSize_y][refSize_x],
		const Matrix4 Ttrack, const Matrix4 view,
		const float dist_threshold,
		const float normal_threshold)
__attribute__((pencil_access(track_core_summary)));

static void reduce_core_summary(float sums[restrict const static 32], TrackData row)
{
	for (int z = 0; z < 32; ++z) {
		const float adjustVal = row.J[z/6];
		const float tempVal = sums[z];
		sums[z] = adjustVal + tempVal;
	}
}
void reduce_core(float sums[restrict const static 32], TrackData row)
__attribute__((pencil_access(reduce_core_summary)));

int mm2meters_pencil(unsigned int outSize_x, unsigned int outSize_y,
		float out[restrict const static outSize_y][outSize_x],
		unsigned int inSize_x, unsigned int inSize_y,
		const unsigned short in[restrict const static inSize_y][inSize_x],
		int ratio)
{
#pragma scop
{
	__pencil_assume(outSize_y < 960);
	__pencil_assume(outSize_x < 1280);
	__pencil_assume(outSize_y % 120 == 0);
	__pencil_assume(outSize_x % 160 == 0);
	__pencil_assume(outSize_x > 0);
	__pencil_assume(outSize_y > 0);
	__pencil_assume(inSize_x > 0);
	__pencil_assume(inSize_y > 0);
	for (unsigned int y = 0; y < outSize_y; y++) {
		for (unsigned int x = 0; x < outSize_x; x++) {
			int xr = x * ratio;
			int yr = y * ratio;
			out[y][x] = in[yr][xr] / 1000.0f;
		}
	}
}
#pragma endscop
	return 0;
}

int bilateralFilter_pencil(int size_x, int size_y,
		float out[restrict const static size_y][size_x],
		const float in[restrict const static size_y][size_x],
		uint2 size, int gaussianS,
		const float gaussian[restrict const static gaussianS],
		float e_d, int r)
{
	#pragma scop
	{
		__pencil_assume(size_y < 960);
		__pencil_assume(size_x < 1280);
		__pencil_assume(size_y % 120 == 0);
		__pencil_assume(size_x % 160 == 0);
		__pencil_assume(size_x > 0);
		__pencil_assume(size_y > 0);
		__pencil_assume(r > 0);
		__pencil_assume(r < 16);

		for (unsigned int y = 0; y < size_y; y++) {
			for (unsigned int x = 0; x < size_x; x++) {
				out[y][x] = bilateralFilter_core(x, y, size_x, size_y, r,
						gaussianS, e_d, in, gaussian);
			}
		}
	}
	#pragma endscop
	return 0;
}

inline void inline_mm2meters_pencil(unsigned int outSize_x, unsigned int outSize_y,
		float out[restrict const static outSize_y][outSize_x],
		unsigned int inSize_x, unsigned int inSize_y,
		const unsigned short in[restrict const static inSize_y][inSize_x],
		int ratio)
{
#pragma scop
{
	__pencil_assume(outSize_y < 960);
	__pencil_assume(outSize_x < 1280);
	__pencil_assume(outSize_y % 120 == 0);
	__pencil_assume(outSize_x % 160 == 0);
	__pencil_assume(outSize_x > 0);
	__pencil_assume(outSize_y > 0);
	__pencil_assume(inSize_x > 0);
	__pencil_assume(inSize_y > 0);
	for (unsigned int y = 0; y < outSize_y; y++) {
		for (unsigned int x = 0; x < outSize_x; x++) {
			int xr = x * ratio;
			int yr = y * ratio;
			out[y][x] = in[yr][xr] / 1000.0f;
		}
	}
}
#pragma endscop

}

inline void inline_bilateralFilter_pencil(int size_x, int size_y,
		float out[restrict const static size_y][size_x],
		const float in[restrict const static size_y][size_x],
		uint2 size, int gaussianS,
		const float gaussian[restrict const static gaussianS],
		float e_d, int r)
{
	#pragma scop
	{
		__pencil_assume(size_y < 960);
		__pencil_assume(size_x < 1280);
		__pencil_assume(size_y % 120 == 0);
		__pencil_assume(size_x % 160 == 0);
		__pencil_assume(size_x > 0);
		__pencil_assume(size_y > 0);
		__pencil_assume(r > 0);
		__pencil_assume(r < 16);

		for (unsigned int y = 0; y < size_y; y++) {
			for (unsigned int x = 0; x < size_x; x++) {
				out[y][x] = bilateralFilter_core(x, y, size_x, size_y, r,
						gaussianS, e_d, in, gaussian);
			}
		}
	}
	#pragma endscop

}

int initVolume_pencil(const unsigned int v_size_x, const unsigned int v_size_y, const unsigned int v_size_z,
		short2 v_data[restrict const static v_size_z][v_size_y][v_size_x],
		const float2 d)
{
#pragma scop
{
	__pencil_assume(v_size_x < 1024);
	__pencil_assume(v_size_y < 1024);
	__pencil_assume(v_size_z < 1024);
	__pencil_assume(v_size_x % 256 == 0);
	__pencil_assume(v_size_y % 256 == 0);
	__pencil_assume(v_size_z % 256 == 0);
	__pencil_assume(v_size_x > 0);
	__pencil_assume(v_size_y > 0);
	__pencil_assume(v_size_z > 0);
	for (unsigned int x = 0; x < v_size_x; x++) {
		for (unsigned int y = 0; y < v_size_y; y++) {
			for (unsigned int z = 0; z < v_size_z; z++) {
				initVolume_core(x, y, z, v_size_x, v_size_y, v_size_z, v_data, (d.x * 32766.0f), d.y);
			}
		}
	}
}
#pragma endscop
return 0;
}

int integrateKernel_pencil(const unsigned int vol_size_x, const unsigned int vol_size_y,
		const unsigned int vol_size_z, const float3 vol_dim,
		short2 vol_data[restrict const static vol_size_z][vol_size_y][vol_size_x],
		unsigned int depthSize_x, unsigned int depthSize_y,
		const float depth[restrict const static depthSize_y][depthSize_x],
		const Matrix4 invTrack, const Matrix4 K,
		const float mu, const float maxweight)
{

  const float3 delta = c_rotate(invTrack,
			make_float3(0, 0, vol_dim.z / vol_size_z));
	const float3 cameraDelta = c_rotate(K, delta);
#pragma scop

	{
		__pencil_assume(vol_size_x < 1024);
		__pencil_assume(vol_size_y < 1024);
		__pencil_assume(vol_size_z < 1024);
		__pencil_assume(vol_size_x % 256 == 0);
		__pencil_assume(vol_size_y % 256 == 0);
		__pencil_assume(vol_size_z % 256 == 0);
		__pencil_assume(vol_size_x > 0);
		__pencil_assume(vol_size_y > 0);
		__pencil_assume(vol_size_z > 0);
		__pencil_assume(depthSize_x > 0);
		__pencil_assume(depthSize_y > 0);
		for (unsigned int y = 0; y < vol_size_y; y++) {
			for (unsigned int x = 0; x < vol_size_x; x++) {
				integrateKernel_core(vol_size_x, vol_size_y, vol_size_z, vol_dim,
						vol_data, x, y, depthSize_x, depthSize_y, depth,
						invTrack, K, mu, maxweight, delta, cameraDelta);
			}
		}
	}
#pragma endscop
	return 0;
}

int depth2vertex_pencil(unsigned int imageSize_x, unsigned int imageSize_y,
		float3 vertex[restrict const static imageSize_y][imageSize_x],
		const float depth[restrict const static imageSize_y][imageSize_x],
		const Matrix4 invK)
{
#pragma scop
	{
		__pencil_assume(imageSize_y < 960);
		__pencil_assume(imageSize_x < 1280);
		__pencil_assume(imageSize_y % 60 == 0);
		__pencil_assume(imageSize_x % 80 == 0);
		__pencil_assume(imageSize_x > 0);
		__pencil_assume(imageSize_y > 0);
		for (unsigned int y = 0; y < imageSize_y; y++) {
			for (unsigned int x = 0; x < imageSize_x; x++) {
				vertex[y][x] = depth2vertex_core(x, y, imageSize_x,
						imageSize_y, depth, invK);
			}
		}
	}
#pragma endscop
	return 0;
}

int vertex2normal_pencil(unsigned int imageSize_x, unsigned int imageSize_y,
		float3 out[restrict const static imageSize_y][imageSize_x],
		const float3 in[restrict const static imageSize_y][imageSize_x])
{
#pragma scop
	{
		__pencil_assume(imageSize_y < 960);
		__pencil_assume(imageSize_x < 1280);
		__pencil_assume(imageSize_y % 60 == 0);
		__pencil_assume(imageSize_x % 80 == 0);
		__pencil_assume(imageSize_x > 0);
		__pencil_assume(imageSize_y > 0);
		for (unsigned int y = 0; y < imageSize_y; y++) {
			for (unsigned int x = 0; x < imageSize_x; x++) {
				out[y][x] = vertex2normal_core(x, y, imageSize_x, imageSize_y, in);
			}
		}
	}
#pragma endscop
	return 0;
}
inline void inline_depth2vertex_pencil(unsigned int imageSize_x, unsigned int imageSize_y,
		float3 vertex[restrict const static imageSize_y][imageSize_x],
		const float depth[restrict const static imageSize_y][imageSize_x],
		const Matrix4 invK)
{
#pragma scop
	{
		__pencil_assume(imageSize_y < 960);
		__pencil_assume(imageSize_x < 1280);
		__pencil_assume(imageSize_y % 60 == 0);
		__pencil_assume(imageSize_x % 80 == 0);
		__pencil_assume(imageSize_x > 0);
		__pencil_assume(imageSize_y > 0);
		for (unsigned int y = 0; y < imageSize_y; y++) {
			for (unsigned int x = 0; x < imageSize_x; x++) {
				vertex[y][x] = depth2vertex_core(x, y, imageSize_x,
						imageSize_y, depth, invK);
			}
		}
	}
#pragma endscop

}

inline void inline_vertex2normal_pencil(unsigned int imageSize_x, unsigned int imageSize_y,
		float3 out[restrict const static imageSize_y][imageSize_x],
		const float3 in[restrict const static imageSize_y][imageSize_x])
{
#pragma scop
	{
		__pencil_assume(imageSize_y < 960);
		__pencil_assume(imageSize_x < 1280);
		__pencil_assume(imageSize_y % 60 == 0);
		__pencil_assume(imageSize_x % 80 == 0);
		__pencil_assume(imageSize_x > 0);
		__pencil_assume(imageSize_y > 0);
		for (unsigned int y = 0; y < imageSize_y; y++) {
			for (unsigned int x = 0; x < imageSize_x; x++) {
				out[y][x] = vertex2normal_core(x, y, imageSize_x, imageSize_y, in);
			}
		}
	}
#pragma endscop

}



int halfSampleRobustImage_pencil(unsigned int outSize_x, unsigned int outSize_y,
		unsigned int inSize_x, unsigned int inSize_y,
		float out[restrict const static outSize_y][outSize_x],
		const float in[restrict const static inSize_y][inSize_x],
		const float e_d, const int r)
{
#pragma scop
	{
		__pencil_assume(outSize_y < 960);
		__pencil_assume(outSize_x < 1280);
		__pencil_assume(outSize_y % 60 == 0);
		__pencil_assume(outSize_x % 80 == 0);
		__pencil_assume(outSize_x > 0);
		__pencil_assume(outSize_y > 0);

		__pencil_assume(inSize_y < 960);
		__pencil_assume(inSize_x < 1280);
		__pencil_assume(inSize_y % 60 == 0);
		__pencil_assume(inSize_x % 80 == 0);
		__pencil_assume(inSize_x > 0);
		__pencil_assume(inSize_y > 0);
		for (unsigned int y = 0; y < outSize_y; y++) {
			for (unsigned int x = 0; x < outSize_x; x++) {
				out[y][x] = halfSampleRobustImage_core(x, y, outSize_x, outSize_y,
						inSize_x, inSize_y, in, e_d, r);
			}
		}
	}
#pragma endscop
return 0;
}
inline void inline_halfSampleRobustImage_pencil(unsigned int outSize_x, unsigned int outSize_y,
		unsigned int inSize_x, unsigned int inSize_y,
		float out[restrict const static outSize_y][outSize_x],
		const float in[restrict const static inSize_y][inSize_x],
		const float e_d, const int r)
{
#pragma scop
	{
		__pencil_assume(outSize_y < 960);
		__pencil_assume(outSize_x < 1280);
		__pencil_assume(outSize_y % 60 == 0);
		__pencil_assume(outSize_x % 80 == 0);
		__pencil_assume(outSize_x > 0);
		__pencil_assume(outSize_y > 0);

		__pencil_assume(inSize_y < 960);
		__pencil_assume(inSize_x < 1280);
		__pencil_assume(inSize_y % 60 == 0);
		__pencil_assume(inSize_x % 80 == 0);
		__pencil_assume(inSize_x > 0);
		__pencil_assume(inSize_y > 0);
		for (unsigned int y = 0; y < outSize_y; y++) {
			for (unsigned int x = 0; x < outSize_x; x++) {
				out[y][x] = halfSampleRobustImage_core(x, y, outSize_x, outSize_y,
						inSize_x, inSize_y, in, e_d, r);
			}
		}
	}
#pragma endscop

}


int renderNormal_pencil(unsigned int normalSize_x, unsigned int normalSize_y,
		uchar3 out[restrict const static normalSize_y][normalSize_x],
		const float3 normal[restrict const static normalSize_y][normalSize_x])
{
#pragma scop
	{
		__pencil_assume(normalSize_y < 960);
		__pencil_assume(normalSize_x < 1280);
		__pencil_assume(normalSize_y % 120 == 0);
		__pencil_assume(normalSize_x % 160 == 0);
		__pencil_assume(normalSize_x > 0);
		__pencil_assume(normalSize_y > 0);
		for (unsigned int y = 0; y < normalSize_y; y++) {
			for (unsigned int x = 0; x < normalSize_x; x++) {
				out[y][x] = renderNormal_core(x, y, normalSize_x, normalSize_y, normal);
			}
		}
	}
#pragma endscop
	return 0;
}

int renderDepth_pencil(unsigned int depthSize_x, unsigned int depthSize_y,
		uchar4 out[restrict const static depthSize_y][depthSize_x],
		const float depth[restrict const static depthSize_y][depthSize_x],
		const float nearPlane, const float farPlane)
{
	float rangeScale = 1 / (farPlane - nearPlane);
#pragma scop
	{
		__pencil_assume(depthSize_y < 960);
		__pencil_assume(depthSize_x < 1280);
		__pencil_assume(depthSize_y % 120 == 0);
		__pencil_assume(depthSize_x % 160 == 0);
		__pencil_assume(depthSize_x > 0);
		__pencil_assume(depthSize_y > 0);
		for (unsigned int y = 0; y < depthSize_y; y++) {
			for (unsigned int x = 0; x < depthSize_x; x++) {
				out[y][x] = renderDepth_core(x, y, depthSize_x, depthSize_y,
						depth, nearPlane, farPlane, rangeScale);
			}
		}
	}
#pragma endscop
	return 0;
}

int renderTrack_pencil(unsigned int outSize_x, unsigned int outSize_y,
		uchar4 out[restrict const static outSize_y][outSize_x],
		const TrackData data[restrict const static outSize_y][outSize_x])
{
#pragma scop
	{
		__pencil_assume(outSize_y < 960);
		__pencil_assume(outSize_x < 1280);
		__pencil_assume(outSize_y % 120 == 0);
		__pencil_assume(outSize_x % 160 == 0);
		__pencil_assume(outSize_x > 0);
		__pencil_assume(outSize_y > 0);
		for (unsigned int y = 0; y < outSize_y; y++) {
			for (unsigned int x = 0; x < outSize_x; x++) {
				out[y][x] = renderTrack_core (x, y, outSize_x, outSize_y, data);
			}
		}
	}
#pragma endscop
	return 0;
}

int renderVolume_pencil(unsigned int depthSize_x, unsigned int depthSize_y,
		uchar4 out[restrict const static depthSize_y][depthSize_x],
		const unsigned int volume_size_x, const unsigned int volume_size_y, const unsigned int volume_size_z,
		const short2 volume_data[restrict const static volume_size_z][ volume_size_y][volume_size_x],
		const float3 volume_dim, const Matrix4 view,
		const float nearPlane, const float farPlane,
		const float step, const float largestep,
		const float3 light, const float3 ambient)
{
#pragma scop
	{
		__pencil_assume(depthSize_y < 960);
		__pencil_assume(depthSize_x < 1280);
		__pencil_assume(depthSize_y % 120 == 0);
		__pencil_assume(depthSize_x % 160 == 0);
		__pencil_assume(depthSize_x > 0);
		__pencil_assume(depthSize_y > 0);
		__pencil_assume(volume_size_x == 256);
		__pencil_assume(volume_size_y == 256);
		__pencil_assume(volume_size_z == 256);

		for (unsigned int y = 0; y < depthSize_y; y++) {
			for (unsigned int x = 0; x < depthSize_x; x++) {
				out[y][x] = renderVolume_core(x, y, volume_size_x, volume_size_y,
						volume_size_z, volume_data, volume_dim,
						view, nearPlane, farPlane, step,
						largestep, light, ambient);
			}
		}
	}
#pragma endscop
return 0;
}

int raycast_pencil(unsigned int inputSize_x, unsigned int inputSize_y,
		float3 vertex[restrict const static inputSize_y][inputSize_x],
		float3 normal[restrict const static inputSize_y][inputSize_x],
		const unsigned int integration_size_x, const unsigned int integration_size_y, const unsigned int integration_size_z,
		const short2 integration_data[restrict const static integration_size_z][integration_size_y][integration_size_x],
		const float3 integration_dim, const Matrix4 view,
		const float nearPlane, const float farPlane,
		const float step, const float largestep)
{
#pragma scop
	{
		__pencil_assume(inputSize_y < 960);
		__pencil_assume(inputSize_x < 1280);
		__pencil_assume(inputSize_y % 120 == 0);
		__pencil_assume(inputSize_x % 160 == 0);
		__pencil_assume(inputSize_x > 0);
		__pencil_assume(inputSize_y > 0);
		__pencil_assume(integration_size_x == 256);
		__pencil_assume(integration_size_y == 256);
		__pencil_assume(integration_size_z == 256);


		for (unsigned int y = 0; y < inputSize_y; y++) {
			for (unsigned int x = 0; x < inputSize_x; x++) {
				raycast_core(x, y, inputSize_x, inputSize_y, vertex, normal,
						integration_size_x, integration_size_y, integration_size_z,
						integration_data, integration_dim, view, nearPlane,
						farPlane, step, largestep);
			}
		}
	}
#pragma endscop
return 0;
}

int track_pencil(unsigned int refSize_x, unsigned int refSize_y, unsigned int inSize_x, unsigned int inSize_y,
		TrackData output[restrict const static refSize_y][refSize_x],
		const float3 inVertex[restrict const static inSize_y][inSize_x],
		const float3 inNormal[restrict const static inSize_y][inSize_x],
		const float3 refVertex[restrict const static refSize_y][refSize_x],
		const float3 refNormal[restrict const static refSize_y][refSize_x],
		const Matrix4 Ttrack, const Matrix4 view,
		const float dist_threshold, const float normal_threshold)
{
#pragma scop
	{
		__pencil_assume(inSize_y < 960);
		__pencil_assume(inSize_x < 1280);
		__pencil_assume(inSize_y % 60 == 0);
		__pencil_assume(inSize_x % 80 == 0);
		__pencil_assume(inSize_x > 0);
		__pencil_assume(inSize_y > 0);
		for (unsigned int y = 0; y < inSize_y; y++) {
			for (unsigned int x = 0; x < inSize_x; x++) {
				output[y][x] = track_core(refSize_x, refSize_y, output[y][x],
						inVertex[y][x], inNormal[y][x],
						refVertex, refNormal, Ttrack, view,
						dist_threshold, normal_threshold);
			}
		}
	}
#pragma endscop
	return 0;
}

int reduce_pencil(float sums[restrict const static 8][32], const unsigned int Jsize_x, const unsigned int Jsize_y,
		TrackData J[restrict const static Jsize_y][Jsize_x],
		const unsigned int size_x, const unsigned int size_y)
{
#pragma scop
	{
		__pencil_assume(size_y < 960);
		__pencil_assume(size_x < 1280);
		__pencil_assume(Jsize_y % 120 == 0);
		__pencil_assume(Jsize_x % 160 == 0);
		__pencil_assume(size_y % 60 == 0);
		__pencil_assume(size_x % 80 == 0);
		__pencil_assume(size_y > 0);
		__pencil_assume(size_x > 0);


		float intrmdSums[size_x][8][32];

		for (unsigned int blockIndex = 0; blockIndex < 8; blockIndex++) {
			for (unsigned int i = 0; i < 32; ++i) {
				sums[blockIndex][i] = 0;
				for (unsigned int x = 0; x < size_x; x++) {
					intrmdSums[x][blockIndex][i] = 0;
				}
			}
		}
		for (unsigned int blockIndex = 0; blockIndex < 8; blockIndex++) {
			for (unsigned int y = blockIndex; y < size_y; y += 8) {
				for (unsigned int x = 0; x < size_x; x++) {
					reduce_core (intrmdSums[x][blockIndex], J[y][x]);
				}
			}
		}
		for (unsigned int blockIndex = 0; blockIndex < 8; blockIndex++) {
			for (unsigned int i = 0; i < 32; ++i) {
				for (unsigned int x = 0; x < size_x; x++) {
					sums[blockIndex][i] += intrmdSums[x][blockIndex][i];
				}
			}
		}
	}
#pragma endscop
	return 0;
}



inline void inline_track_pencil(unsigned int refSize_x, unsigned int refSize_y, unsigned int inSize_x, unsigned int inSize_y,
		TrackData output[restrict const static refSize_y][refSize_x],
		const float3 inVertex[restrict const static inSize_y][inSize_x],
		const float3 inNormal[restrict const static inSize_y][inSize_x],
		const float3 refVertex[restrict const static refSize_y][refSize_x],
		const float3 refNormal[restrict const static refSize_y][refSize_x],
		const Matrix4 Ttrack, const Matrix4 view,
		const float dist_threshold, const float normal_threshold)
{
#pragma scop
	{
		__pencil_assume(inSize_y < 960);
		__pencil_assume(inSize_x < 1280);
		__pencil_assume(inSize_y % 60 == 0);
		__pencil_assume(inSize_x % 80 == 0);
		__pencil_assume(inSize_x > 0);
		__pencil_assume(inSize_y > 0);
		for (unsigned int y = 0; y < inSize_y; y++) {
			for (unsigned int x = 0; x < inSize_x; x++) {
				output[y][x] = track_core(refSize_x, refSize_y, output[y][x],
						inVertex[y][x], inNormal[y][x],
						refVertex, refNormal, Ttrack, view,
						dist_threshold, normal_threshold);
			}
		}
	}
#pragma endscop

}

inline void inline_reduce_pencil(float sums[restrict const static 8][32], const unsigned int Jsize_x, const unsigned int Jsize_y,
		TrackData J[restrict const static Jsize_y][Jsize_x],
		const unsigned int size_x, const unsigned int size_y)
{
#pragma scop
	{
		__pencil_assume(size_y < 960);
		__pencil_assume(size_x < 1280);
		__pencil_assume(Jsize_y % 120 == 0);
		__pencil_assume(Jsize_x % 160 == 0);
		__pencil_assume(size_y % 60 == 0);
		__pencil_assume(size_x % 80 == 0);
		__pencil_assume(size_y > 0);
		__pencil_assume(size_x > 0);


		float intrmdSums[size_x][8][32];

		for (unsigned int blockIndex = 0; blockIndex < 8; blockIndex++) {
			for (unsigned int i = 0; i < 32; ++i) {
				sums[blockIndex][i] = 0;
				for (unsigned int x = 0; x < size_x; x++) {
					intrmdSums[x][blockIndex][i] = 0;
				}
			}
		}
		for (unsigned int blockIndex = 0; blockIndex < 8; blockIndex++) {
			for (unsigned int y = blockIndex; y < size_y; y += 8) {
				for (unsigned int x = 0; x < size_x; x++) {
					reduce_core (intrmdSums[x][blockIndex], J[y][x]);
				}
			}
		}
		for (unsigned int blockIndex = 0; blockIndex < 8; blockIndex++) {
			for (unsigned int i = 0; i < 32; ++i) {
				for (unsigned int x = 0; x < size_x; x++) {
					sums[blockIndex][i] += intrmdSums[x][blockIndex][i];
				}
			}
		}
 

		// Bruno : Add final reduction
		
		for (int j = 1; j < 8; ++j) {
		  for (int x = 0; x < 32; ++x) {
		    sums [0] [1 + x] += sums [j] [1 + x ];
		  }
		}
		
		for (int i = 0; i < 32; i++) {
		  sums[0][i] = sums[0][i+1];
		}
		
	}
#pragma endscop

}


 void inline_original_update_pose_pencil(Matrix4 pose , float output[restrict const static 32] ) {
  
    float mU[6][6];


    for (int r = 1; r < 6; ++r)
        for (int c = 0; c < r; ++c)
            mU[r][c] = 0;



    mU[0][0] = output[7];
    mU[0][1] = output[8];
    mU[0][2] = output[9];
    mU[0][3] = output[10];
    mU[0][4] = output[11];
    mU[0][5] = output[12];
    mU[1][1] = output[13];
    mU[1][2] = output[14];
    mU[1][3] = output[15];
    mU[1][4] = output[16];
    mU[1][5] = output[17];
    mU[2][2] = output[18];
    mU[2][3] = output[19];
    mU[2][4] = output[20];
    mU[2][5] = output[21];
    mU[3][3] = output[22];
    mU[3][4] = output[23];
    mU[3][5] = output[24];
    mU[4][4] = output[25];
    mU[4][5] = output[26];
    mU[5][5] = output[27];

    for (int r = 1; r < 6; ++r)
        for (int c = 0; c < r; ++c)
            mU[r][c] = mU[c][r];


    int nError = 0;



















    //Bidiagonalize();

    float vDiagonal[6];
    float vOffDiagonal[6];
    float mV[6][6];

     // ------------  Householder reduction to bidiagonal form
     float g = 0.0;
     float scale = 0.0;
     float anorm = 0.0;
     for(int i=0; i<6; ++i) // 300
       {
     const int l = i+1;
     vOffDiagonal[i] = scale * g;
     g = 0.0;
     float s = 0.0;
     scale = 0.0;
     if( i < 6 )
       {
         for(int k=i; k<6; ++k)
           scale += fabs(mU[k][i]);
         if(scale != 0.0)
           {
         for(int k=i; k<6; ++k)
           {
             mU[k][i] /= scale;
             s += mU[k][i] * mU[k][i];
           }
         float f = mU[i][i];
         g = -(f>=0?sqrt(s):-sqrt(s));
         float h = f * g - s;
         mU[i][i] = f - g;
         if(i!=(6-1))
           {
             for(int j=l; j<6; ++j)
               {
             s = 0.0;
             for(int k=i; k<6; ++k)
               s += mU[k][i] * mU[k][j];
             f = s / h;
             for(int k=i; k<6; ++k)
               mU[k][j] += f * mU[k][i];
               } // 150
           }// 190
         for(int k=i; k<6; ++k)
           mU[k][i] *= scale;
           } // 210
       } // 210
     vDiagonal[i] = scale * g;
     g = 0.0;
     s = 0.0;
     scale = 0.0;
     if(!(i >= 6 || i == (6-1)))
       {
         for(int k=l; k<6; ++k)
           scale += fabs(mU[i][k]);
         if(scale != 0.0)
           {
         for(int k=l; k<6; k++)
           {
             mU[i][k] /= scale;
             s += mU[i][k] * mU[i][k];
           }
         float f = mU[i][l];
         g = -(f>=0?sqrt(s):-sqrt(s));
         float h = f * g - s;
         mU[i][l] = f - g;
         for(int k=l; k<6; ++k)
           vOffDiagonal[k] = mU[i][k] / h;
         if(i != (6-1)) // 270
           {
             for(int j=l; j<6; ++j)
               {
             s = 0.0;
             for(int k=l; k<6; ++k)
               s += mU[j][k] * mU[i][k];
             for(int k=l; k<6; ++k)
               mU[j][k] += s * vOffDiagonal[k];
               } // 260
           } // 270
         for(int k=l; k<6; ++k)
           mU[i][k] *= scale;
           } // 290
       } // 290
     anorm = max(anorm, fabs(vDiagonal[i]) + fabs(vOffDiagonal[i]));
       } // 300

     // Accumulation of right-hand transformations












    //Accumulate_RHS();

     // Get rid of fakey loop over ii, do a loop over i directly
       // This here would happen on the first run of the loop with
       // i = N-1
       mV[6-1][6-1] = 1;
       float gbis = vOffDiagonal[6-1];

       // The loop
       for(int i=6-2; i>=0; --i) // 400
         {
       const int l = i + 1;
       if( gbis!=0) // 360
         {
           for(int j=l; j<6; ++j)
             mV[j][i] = (mU[i][j] / mU[i][l]) / gbis; // float division avoids possible underflow
           for(int j=l; j<6; ++j)
             { // 350
               float s = 0;
           for(int k=l; k<6; ++k)
             s += mU[i][k] * mV[k][j];
           for(int k=l; k<6; ++k)
             mV[k][j] += s * mV[k][i];
             } // 350
         } // 360
       for(int j=l; j<6; ++j)
         mV[i][j] = mV[j][i] = 0;
       mV[i][i] =1;
       gbis = vOffDiagonal[i];
         } // 400





    //Accumulate_LHS();

       // Same thing; remove loop over dummy ii and do straight over i
         // Some implementations start from N here
         for(int i=6-1; i>=0; --i)
           { // 500
         const int l = i+1;
         float g = vDiagonal[i];
         // SqSVD here uses i<N ?
         if(i != (6-1))
           for(int j=l; j<6; ++j)
             mU[i][j] = 0.0;
         if(g == 0.0)
           for(int j=i; j<6; ++j)
             mU[j][i] = 0.0;
         else
           { // 475
             // Can pre-divide g here
             float inv_g = 1 / g;
             if(i != (6-1))
               { // 460
             for(int j=l; j<6; ++j)
               { // 450
                 float s = 0;
                 for(int k=l; k<6; ++k)
                   s += mU[k][i] * mU[k][j];
                 float f = (s / mU[i][i]) * inv_g;  // float division
                 for(int k=i; k<6; ++k)
                   mU[k][j] += f * mU[k][i];
               } // 450
               } // 460
             for(int j=i; j<6; ++j)
               mU[j][i] *= inv_g;
           } // 475
         mU[i][i] += 1;
           } // 500


    //Diagonalize();

         // Loop directly over descending k
           for(int k=6-1; k>=0; --k)
             { // 700
           int nIterations = 0;
           float z;
           int bConverged_Or_Error = 0;
	   int first = 1;
           while(!bConverged_Or_Error || first == 1)
           {
	     first = 0;
               int result = 0;
             //  bConverged_Or_Error = Diagonalize_SubLoop(k, z);

                const int k1 = k-1;
                // 520 is here!
                for(int l=k; l>=0; --l)
                  { // 530
                const int l1 = l-1;
                if((fabs(vOffDiagonal[l]) + anorm) == anorm)
                    goto line_565;
                if((fabs(vDiagonal[l1]) + anorm) == anorm) {
                    goto line_540;
		}
                continue;

                line_540:
                  {
                    float c = 0;
                    float s = 1.0;
                    for(int i=l; i<=k; ++i)
                      { // 560
                    float f = s * vOffDiagonal[i];
                    vOffDiagonal[i] *= c;
                    if((fabs(f) + anorm) == anorm)
                      break; // goto 565, effectively
                    float g = vDiagonal[i];
                    float h = sqrt(f * f + g * g); // Other implementations do this bit better
                    vDiagonal[i] = h;
                    c = g / h;
                    s = -f / h;
                    if(1)
                      for(int j=0; j<6; ++j)
                        {
                          float y = mU[j][l1];
                          float z = mU[j][i];
                          mU[j][l1] = y*c + z*s;
                          mU[j][i] = -y*s + z*c;
                        }
                      } // 560
                  }

                line_565:
                  {
                    // Check for convergence..
                    z = vDiagonal[k];
                    if(l == k) {
                        result = 1; // convergence.
                        goto line_end_of_do;
                    }
                    if(nIterations == 30)
                      {
                    nError = k;
                    result = 1; // convergence.
                    goto line_end_of_do;
                      }
                    ++nIterations;
                    float x = vDiagonal[l];
                    float y = vDiagonal[k1];
                    float g = vOffDiagonal[k1];
                    float h = vOffDiagonal[k];
                    float f = ((y-z)*(y+z) + (g-h)*(g+h)) / (2.0*h*y);
                    g = sqrt(f*f + 1.0);
                    float signed_g =  (f>=0)?g:-g;
                    f = ((x-z)*(x+z) + h*(y/(f + signed_g) - h)) / x;

                    // Next QR transformation
                    float c = 1.0;
                    float s = 1.0;
                    for(int i1 = l; i1<=k1; ++i1)
                      { // 600
                    const int i=i1+1;
                    g = vOffDiagonal[i];
                    y = vDiagonal[i];
                    h = s*g;
                    g = c*g;
                    z = sqrt(f*f + h*h);
                    vOffDiagonal[i1] = z;
                    c = f/z;
                    s = h/z;
                    f = x*c + g*s;
                    g = -x*s + g*c;
                    h = y*s;
                    y *= c;
                    if(1)
                      for(int j=0; j<6; ++j)
                        {
                          float xx = mV[j][i1];
                          float zz = mV[j][i];
                          mV[j][i1] = xx*c + zz*s;
                          mV[j][i] = -xx*s + zz*c;
                        }
                    z = sqrt(f*f + h*h);
                    vDiagonal[i1] = z;
                    if(z!=0)
                      {
                        c = f/z;
                        s = h/z;
                      }
                    f = c*g + s*y;
                    x = -s*g + c*y;
                    if(1)
                      for(int j=0; j<6; ++j)
                        {
                          float yy = mU[j][i1];
                          float zz = mU[j][i];
                          mU[j][i1] = yy*c + zz*s;
                          mU[j][i] = -yy*s + zz*c;
                        }
                      } // 600
                    vOffDiagonal[l] = 0;
                    vOffDiagonal[k] = f;
                    vDiagonal[k] = x;
                    result = 0; // convergence.
                    goto line_end_of_do;
                    // EO IF NOT CONVERGED CHUNK
                  } // EO IF TESTS HOLD
                  } // 530
                // Code should never get here!

                line_end_of_do :
                bConverged_Or_Error = result;
           }


           if(nError) {
             return;
           }

           if(z < 0)
             {
               vDiagonal[k] = -z;
               if(1)
                 for(int j=0; j<6; ++j)
                     mV[j][k] = -mV[j][k];
             }
             } // 700




    float inv_diag[6];

    float dMax = vDiagonal[0];
    for(int i=1; i<6; ++i) dMax = max(dMax, vDiagonal[i]);

    for(int i=0; i<6; ++i)
        inv_diag[i] = (vDiagonal[i] * 1e6 > dMax) ? 1/vDiagonal[i] : 0;


    float b[6];
    b[0] = output[1];
    b[1] = output[2];
    b[2] = output[3];
    b[3] = output[4];
    b[4] = output[5];
    b[5] = output[6];

    // Transpose mU
    float TmU[6][6];
    for(int i=0; i<6; ++i)
        for(int j=0; j<6; ++j) {
            TmU[j][i] = mU[i][j] ;
        }




    float vTmUfvb[6];// = vTmU * vb;

    for(int i=0; i<6; i++) {
        vTmUfvb[i] = 0;
        for(int k = 0; k<6; k++)
            vTmUfvb[i] += TmU[i][k]*b[k];
    }

    float diagmultres [6];//= diagmult(vinv_diag, vTmUfvb) ;

    for(int i=0; i<6; i++) {
        diagmultres[i]  = inv_diag[i]*vTmUfvb[i];
    }


    float x [6];

    for(int i=0; i<6; i++) {
        x[i] = 0;
        for(int k = 0; k<6; k++)
            x[i] += mV[i][k]*diagmultres[k];
    }


    // From here only MatMult, Ok

        const float one_6th = 1.0/6.0;
        const float one_20th = 1.0/20.0;


        float  my_rotation_matrix[3][3];
        float my_translation[3];


        float w[3];
        w[0] = x[3];
        w[1] = x[4];
        w[2] = x[5];
        float xf[3];
        xf[0] = x[0];
        xf[1] = x[1];
        xf[2] = x[2];
        const float theta_sq = w[0]*w[0] + w[1]*w[1] + w[2]*w[2] ;
        const float theta = sqrt(theta_sq);
        float A, B;

       float cross[3];

        cross[0] = w[1]*xf[2] - w[2]*xf[1];
        cross[1] = w[2]*xf[0] - w[0]*xf[2];
        cross[2] = w[0]*xf[1] - w[1]*xf[0];

        if (theta_sq < 1e-8) {
            A = 1.0 - one_6th * theta_sq;
            B = 0.5;
            my_translation[0] = xf[0] + 0.5 * cross[0];
            my_translation[1] = xf[1] + 0.5 * cross[1];
            my_translation[2] = xf[2] + 0.5 * cross[2];
        } else {
            float C;
            if (theta_sq < 1e-6) {
                C = one_6th*(1.0 - one_20th * theta_sq);
                A = 1.0 - theta_sq * C;
                B = 0.5 - 0.25 * one_6th * theta_sq;
            } else {
                const float inv_theta = 1.0/theta;
                A = sin(theta) * inv_theta;
                B = (1 - cos(theta)) * (inv_theta * inv_theta);
                C = (1 - A) * (inv_theta * inv_theta);
            }

            float wcross[3];

            wcross[0] = w[1]*cross[2] - w[2]*cross[1];
            wcross[1] = w[2]*cross[0] - w[0]*cross[2];
            wcross[2] = w[0]*cross[1] - w[1]*cross[0];

            float Bcross[3];
            Bcross[0] = B *  cross[0];
            Bcross[1] = B *  cross[1];
            Bcross[2] = B *  cross[2];

            float Ccross[3];
            Ccross[0] = C *  cross[0];
            Ccross[1] = C *  cross[1];
            Ccross[2] = C *  cross[2];


            my_translation[0] = xf[0] + Bcross[0] + Ccross[0];
            my_translation[1] = xf[1] + Bcross[1] + Ccross[1];
            my_translation[2] = xf[2] + Bcross[2] + Ccross[2];
        }

        //rodrigues_so3_exp(w, A, B, result.get_rotation().my_matrix);
        {
            const float wx2 = (float)w[0]*w[0];
            const float wy2 = (float)w[1]*w[1];
            const float wz2 = (float)w[2]*w[2];

            my_rotation_matrix[0][0] = 1.0 - B*(wy2 + wz2);
            my_rotation_matrix[1][1] = 1.0 - B*(wx2 + wz2);
            my_rotation_matrix[2][2] = 1.0 - B*(wx2 + wy2);
        }
        {
            const float a = A*w[2];
            const float b = B*(w[0]*w[1]);
            my_rotation_matrix[0][1] = b - a;
            my_rotation_matrix[1][0] = b + a;
        }
        {
            const float a = A*w[1];
            const float b = B*(w[0]*w[2]);
            my_rotation_matrix[0][2] = b + a;
            my_rotation_matrix[2][0] = b - a;
        }
        {
            const float a = A*w[0];
            const float b = B*(w[1]*w[2]);
            my_rotation_matrix[1][2] = b - a;
            my_rotation_matrix[2][1] = b + a;
        }

        /*
        my_rotation_matrix[0] = myunit(my_rotation_matrix[0]);
        my_rotation_matrix[1] -= my_rotation_matrix[0] * (my_rotation_matrix[0]*my_rotation_matrix[1]);
        my_rotation_matrix[1] = myunit(my_rotation_matrix[1]);
        my_rotation_matrix[2] -= my_rotation_matrix[0] * (my_rotation_matrix[0]*my_rotation_matrix[2]);
        my_rotation_matrix[2] -= my_rotation_matrix[1] * (my_rotation_matrix[1]*my_rotation_matrix[2]);
        my_rotation_matrix[2] = myunit(my_rotation_matrix[2]);
        */

        {
            float vv = my_rotation_matrix[0][0] *  my_rotation_matrix[0][0] +my_rotation_matrix[0][1] *  my_rotation_matrix[0][1] +my_rotation_matrix[0][2] *  my_rotation_matrix[0][2] ;
            float coef = (1/sqrt(vv));
            my_rotation_matrix[0][0] = my_rotation_matrix[0][0] * coef;
            my_rotation_matrix[0][1] = my_rotation_matrix[0][1] * coef;
            my_rotation_matrix[0][2] = my_rotation_matrix[0][2] * coef;
        }

        {

        float my_rotation_matrix01 =  my_rotation_matrix[0][0] *  my_rotation_matrix[1][0]
                                     +my_rotation_matrix[0][1] *  my_rotation_matrix[1][1]
                                     +my_rotation_matrix[0][2] *  my_rotation_matrix[1][2] ;

        float my_rotation_matrix001[3];
        my_rotation_matrix001[0] = my_rotation_matrix[0][0] * my_rotation_matrix01;
        my_rotation_matrix001[1] = my_rotation_matrix[0][1] * my_rotation_matrix01;
        my_rotation_matrix001[2] = my_rotation_matrix[0][2] * my_rotation_matrix01;

        my_rotation_matrix[1][0] -= my_rotation_matrix001[0];
        my_rotation_matrix[1][1] -= my_rotation_matrix001[1];
        my_rotation_matrix[1][2] -= my_rotation_matrix001[2];

        }



        {
            float vv = my_rotation_matrix[1][0] *  my_rotation_matrix[1][0] +my_rotation_matrix[1][1] *  my_rotation_matrix[1][1] +my_rotation_matrix[1][2] *  my_rotation_matrix[1][2] ;
            float coef = (1/sqrt(vv));
            my_rotation_matrix[1][0] = my_rotation_matrix[1][0] * coef;
            my_rotation_matrix[1][1] = my_rotation_matrix[1][1] * coef;
            my_rotation_matrix[1][2] = my_rotation_matrix[1][2] * coef;
        }

        {

        float my_rotation_matrix02 =  my_rotation_matrix[0][0] *  my_rotation_matrix[2][0]
                                     +my_rotation_matrix[0][1] *  my_rotation_matrix[2][1]
                                     +my_rotation_matrix[0][2] *  my_rotation_matrix[2][2] ;

        float my_rotation_matrix002[3];
        my_rotation_matrix002[0] = my_rotation_matrix[0][0] * my_rotation_matrix02;
        my_rotation_matrix002[1] = my_rotation_matrix[0][1] * my_rotation_matrix02;
        my_rotation_matrix002[2] = my_rotation_matrix[0][2] * my_rotation_matrix02;

        my_rotation_matrix[2][0] -= my_rotation_matrix002[0];
        my_rotation_matrix[2][1] -= my_rotation_matrix002[1];
        my_rotation_matrix[2][2] -= my_rotation_matrix002[2];

        }

        {

        float my_rotation_matrix12 =  my_rotation_matrix[1][0] *  my_rotation_matrix[2][0]
                                     +my_rotation_matrix[1][1] *  my_rotation_matrix[2][1]
                                     +my_rotation_matrix[1][2] *  my_rotation_matrix[2][2] ;

        float my_rotation_matrix112[3];
        my_rotation_matrix112[0] = my_rotation_matrix[1][0] * my_rotation_matrix12;
        my_rotation_matrix112[1] = my_rotation_matrix[1][1] * my_rotation_matrix12;
        my_rotation_matrix112[2] = my_rotation_matrix[1][2] * my_rotation_matrix12;

        my_rotation_matrix[2][0] -= my_rotation_matrix112[0];
        my_rotation_matrix[2][1] -= my_rotation_matrix112[1];
        my_rotation_matrix[2][2] -= my_rotation_matrix112[2];

        }




        {
            float vv = my_rotation_matrix[2][0] *  my_rotation_matrix[2][0] +my_rotation_matrix[2][1] *  my_rotation_matrix[2][1] +my_rotation_matrix[2][2] *  my_rotation_matrix[2][2] ;
            float coef = (1/sqrt(vv));
            my_rotation_matrix[2][0] = my_rotation_matrix[2][0] * coef;
            my_rotation_matrix[2][1] = my_rotation_matrix[2][1] * coef;
            my_rotation_matrix[2][2] = my_rotation_matrix[2][2] * coef;
        }





        float I [4][4];
        for (int i = 0 ; i < 4 ; i++) {
            for (int j = 0 ; j < 4 ; j++) {
                I[i][j] = 0;
            }
        }
        for (int i = 0 ; i < 4 ; i++) {
            I[i][i] = 1;
        }


    float TmpRes[4][4];
    float Itranspose[4][4];
    for(int i=0; i<4; ++i)
           for(int j=0; j<4; ++j) {
               Itranspose[j][i] = I[i][j] ;
           }

    for(int i=0; i<4; ++i) {
        //TmpRes[i].slice<0,3>()=my_rotation_matrix *  Itranspose[i].slice<0,3>();

        TmpRes[i][0] = my_rotation_matrix[0][0] *  Itranspose[i][0]
                     + my_rotation_matrix[0][1] *  Itranspose[i][1]
                     + my_rotation_matrix[0][2] *  Itranspose[i][2] ;
        TmpRes[i][1] = my_rotation_matrix[1][0] *  Itranspose[i][0]
                     + my_rotation_matrix[1][1] *  Itranspose[i][1]
                     + my_rotation_matrix[1][2] *  Itranspose[i][2] ;
        TmpRes[i][2] = my_rotation_matrix[2][0] *  Itranspose[i][0]
                     + my_rotation_matrix[2][1] *  Itranspose[i][1]
                     + my_rotation_matrix[2][2] *  Itranspose[i][2] ;

        TmpRes[i][0] += my_translation[0] *  Itranspose[i][3];
        TmpRes[i][1] += my_translation[1] *  Itranspose[i][3];
        TmpRes[i][2] += my_translation[2] *  Itranspose[i][3];
        TmpRes[i][3] =  Itranspose[i][3];
    }
    Matrix4 RR;

    for(int j=0; j<4; ++j) {
        RR.data[j].x =  TmpRes[0][j] ;
        RR.data[j].y =  TmpRes[1][j] ;
        RR.data[j].z =  TmpRes[2][j] ;
        RR.data[j].w =  TmpRes[3][j] ;

    }
    Matrix4 m1 = pose;
    Matrix4 m2 = RR;
    Matrix4 m3;
    for (int i = 0; i < 4; i++ ) {
      m3.data[ i ].x = 0;
      m3.data[ i ].x += m1.data[ i ].x * m2.data[ 0 ].x;
      m3.data[ i ].x += m1.data[ i ].y * m2.data[ 1 ].x;
      m3.data[ i ].x += m1.data[ i ].z * m2.data[ 2 ].x;
      m3.data[ i ].x += m1.data[ i ].w * m2.data[ 3 ].x;
      
      m3.data[ i ].y = 0;
      m3.data[ i ].y += m1.data[ i ].x * m2.data[ 0 ].y;
      m3.data[ i ].y += m1.data[ i ].y * m2.data[ 1 ].y;
      m3.data[ i ].y += m1.data[ i ].z * m2.data[ 2 ].y;
      m3.data[ i ].y += m1.data[ i ].w * m2.data[ 3 ].y;
      
      m3.data[ i ].z = 0;
      m3.data[ i ].z += m1.data[ i ].x * m2.data[ 0 ].z;
      m3.data[ i ].z += m1.data[ i ].y * m2.data[ 1 ].z;
      m3.data[ i ].z += m1.data[ i ].z * m2.data[ 2 ].z;
      m3.data[ i ].z += m1.data[ i ].w * m2.data[ 3 ].z;
      
      m3.data[ i ].w = 0;
      m3.data[ i ].w += m1.data[ i ].x * m2.data[ 0 ].w;
      m3.data[ i ].w += m1.data[ i ].y * m2.data[ 1 ].w;
      m3.data[ i ].w += m1.data[ i ].z * m2.data[ 2 ].w;
      m3.data[ i ].w += m1.data[ i ].w * m2.data[ 3 ].w;
    }

    //pose->data[0] = m3.data[0];
    //pose->data[1] = m3.data[1];
    //pose->data[2] = m3.data[2];
    //pose->data[3] = m3.data[3];

    // Return validity test result of the tracking
    float xsqr = 0;
    for(int i=0; i<6; ++i) {
        xsqr += x[i] * x[i];
    }
    float lnorm =     sqrt(xsqr);

    /* skipped test : lnorm < icp_threshold */

}



void line_540 (int l , int k, int l1, float anorm, float vOffDiagonal[6], float vDiagonal[6], float mU[6][6])  {
  float c = 0;
  float s = 1.0;
  for(int i=l; i<=k; ++i) { // 560
    float f = s * vOffDiagonal[i];
    vOffDiagonal[i] *= c;
    if((fabs(f) + anorm) == anorm)
      break; // goto 565, effectively
    float g = vDiagonal[i];
    float h = sqrt(f * f + g * g); // Other implementations do this bit better
    vDiagonal[i] = h;
    c = g / h;
    s = -f / h;
    if(1)
      for(int j=0; j<6; ++j)
	{
	  float y = mU[j][l1];
	  float z = mU[j][i];
	  mU[j][l1] = y*c + z*s;
	  mU[j][i] = -y*s + z*c;
	}
  } // 560
}


void inline_update_pose_pencil(Matrix4 pose , float output[restrict const static 32] ) {
  
  float mU[6][6];
  
  
  for (int r = 1; r < 6; ++r)
    for (int c = 0; c < r; ++c)
      mU[r][c] = 0;
  


  mU[0][0] = output[7];
  mU[0][1] = output[8];
  mU[0][2] = output[9];
  mU[0][3] = output[10];
  mU[0][4] = output[11];
  mU[0][5] = output[12];
  mU[1][1] = output[13];
  mU[1][2] = output[14];
  mU[1][3] = output[15];
  mU[1][4] = output[16];
  mU[1][5] = output[17];
  mU[2][2] = output[18];
  mU[2][3] = output[19];
  mU[2][4] = output[20];
  mU[2][5] = output[21];
  mU[3][3] = output[22];
  mU[3][4] = output[23];
  mU[3][5] = output[24];
  mU[4][4] = output[25];
  mU[4][5] = output[26];
  mU[5][5] = output[27];

  for (int r = 1; r < 6; ++r)
    for (int c = 0; c < r; ++c)
      mU[r][c] = mU[c][r];


  int nError = 0;

  //Bidiagonalize();

  float vDiagonal[6];
  float vOffDiagonal[6];
  float mV[6][6];

  // ------------  Householder reduction to bidiagonal form
  float g = 0.0;
  float scale = 0.0;
  float anorm = 0.0;
  for(int i=0; i<6; ++i) // 300
    {
      const int l = i+1;
      vOffDiagonal[i] = scale * g;
      g = 0.0;
      float s = 0.0;
      scale = 0.0;
      if( i < 6 )
	{
	  for(int k=i; k<6; ++k)
	    scale += fabs(mU[k][i]);
	  if(scale != 0.0)
	    {
	      for(int k=i; k<6; ++k)
		{
		  mU[k][i] /= scale;
		  s += mU[k][i] * mU[k][i];
		}
	      float f = mU[i][i];
	      g = -(f>=0?sqrt(s):-sqrt(s));
	      float h = f * g - s;
	      mU[i][i] = f - g;
	      if(i!=(6-1))
		{
		  for(int j=l; j<6; ++j)
		    {
		      s = 0.0;
		      for(int k=i; k<6; ++k)
			s += mU[k][i] * mU[k][j];
		      f = s / h;
		      for(int k=i; k<6; ++k)
			mU[k][j] += f * mU[k][i];
		    } // 150
		}// 190
	      for(int k=i; k<6; ++k)
		mU[k][i] *= scale;
	    } // 210
	} // 210
      vDiagonal[i] = scale * g;
      g = 0.0;
      s = 0.0;
      scale = 0.0;
      if(!(i >= 6 || i == (6-1)))
	{
	  for(int k=l; k<6; ++k)
	    scale += fabs(mU[i][k]);
	  if(scale != 0.0)
	    {
	      for(int k=l; k<6; k++)
		{
		  mU[i][k] /= scale;
		  s += mU[i][k] * mU[i][k];
		}
	      float f = mU[i][l];
	      g = -(f>=0?sqrt(s):-sqrt(s));
	      float h = f * g - s;
	      mU[i][l] = f - g;
	      for(int k=l; k<6; ++k)
		vOffDiagonal[k] = mU[i][k] / h;
	      if(i != (6-1)) // 270
		{
		  for(int j=l; j<6; ++j)
		    {
		      s = 0.0;
		      for(int k=l; k<6; ++k)
			s += mU[j][k] * mU[i][k];
		      for(int k=l; k<6; ++k)
			mU[j][k] += s * vOffDiagonal[k];
		    } // 260
		} // 270
	      for(int k=l; k<6; ++k)
		mU[i][k] *= scale;
	    } // 290
	} // 290
      anorm = max(anorm, fabs(vDiagonal[i]) + fabs(vOffDiagonal[i]));
    } // 300

  // Accumulation of right-hand transformations

  //Accumulate_RHS();

  // Get rid of fakey loop over ii, do a loop over i directly
  // This here would happen on the first run of the loop with
  // i = N-1
  mV[6-1][6-1] = 1;
  float gbis = vOffDiagonal[6-1];

  // The loop
  for(int i=6-2; i>=0; --i) {  // 400
    
    const int l = i + 1;
    if( gbis!=0) // 360
      {
	for(int j=l; j<6; ++j)
	  mV[j][i] = (mU[i][j] / mU[i][l]) / gbis; // float division avoids possible underflow
	for(int j=l; j<6; ++j)
	  { // 350
	    float s = 0;
	    for(int k=l; k<6; ++k)
	      s += mU[i][k] * mV[k][j];
	    for(int k=l; k<6; ++k)
	      mV[k][j] += s * mV[k][i];
	  } // 350
      } // 360
    for(int j=l; j<6; ++j)
      mV[i][j] = mV[j][i] = 0;
    mV[i][i] =1;
    gbis = vOffDiagonal[i];
  } // 400





  //Accumulate_LHS();

  // Same thing; remove loop over dummy ii and do straight over i
  // Some implementations start from N here
  for(int i=6-1; i>=0; --i)
    { // 500
      const int l = i+1;
      float g = vDiagonal[i];
      // SqSVD here uses i<N ?
      if(i != (6-1))
	for(int j=l; j<6; ++j)
	  mU[i][j] = 0.0;
      if(g == 0.0)
	for(int j=i; j<6; ++j)
	  mU[j][i] = 0.0;
      else
	{ // 475
	  // Can pre-divide g here
	  float inv_g = 1 / g;
	  if(i != (6-1))
	    { // 460
	      for(int j=l; j<6; ++j)
		{ // 450
		  float s = 0;
		  for(int k=l; k<6; ++k)
		    s += mU[k][i] * mU[k][j];
		  float f = (s / mU[i][i]) * inv_g;  // float division
		  for(int k=i; k<6; ++k)
		    mU[k][j] += f * mU[k][i];
		} // 450
	    } // 460
	  for(int j=i; j<6; ++j)
	    mU[j][i] *= inv_g;
	} // 475
      mU[i][i] += 1;
    } // 500


  //Diagonalize();
  int anticipate_exit = 0;
  // Loop directly over descending k
  for(int k=6-1; k>=0; --k)
    { // 700
      int nIterations = 0;
      float z;
      int bConverged_Or_Error = 0;
      int first = 1;
      while(!bConverged_Or_Error || first == 1)
	{
	  first = 0;
	  int result = 0;
	  //  bConverged_Or_Error = Diagonalize_SubLoop(k, z);

	  const int k1 = k-1;
	  // 520 is here!

	  for(int l=k; l>=0; --l)
	    { // 530
	      int do_565 = 0;
	      const int l1 = l-1;
	      if((fabs(vOffDiagonal[l]) + anorm) == anorm) {
		do_565 = 1;
		
	      } else  if((fabs(vDiagonal[l1]) + anorm) == anorm) {
		line_540 ( l ,  k,  l1,  anorm,  vOffDiagonal,  vDiagonal,  mU) ;
		do_565 = 1;	
	      }
	    
	      if (!do_565) {continue;}

	      {
		// Check for convergence..
		z = vDiagonal[k];
		if(l == k) {
		  result = 1; // convergence.
		  break;
		}
		if(nIterations == 30)
		  {
                    nError = k;
                    result = 1; // convergence.
                    break;
		  }
		++nIterations;
		float x = vDiagonal[l];
		float y = vDiagonal[k1];
		float g = vOffDiagonal[k1];
		float h = vOffDiagonal[k];
		float f = ((y-z)*(y+z) + (g-h)*(g+h)) / (2.0*h*y);
		g = sqrt(f*f + 1.0);
		float signed_g =  (f>=0)?g:-g;
		f = ((x-z)*(x+z) + h*(y/(f + signed_g) - h)) / x;

		// Next QR transformation
		float c = 1.0;
		float s = 1.0;
		for(int i1 = l; i1<=k1; ++i1)
		  { // 600
                    const int i=i1+1;
                    g = vOffDiagonal[i];
                    y = vDiagonal[i];
                    h = s*g;
                    g = c*g;
                    z = sqrt(f*f + h*h);
                    vOffDiagonal[i1] = z;
                    c = f/z;
                    s = h/z;
                    f = x*c + g*s;
                    g = -x*s + g*c;
                    h = y*s;
                    y *= c;
                    if(1)
                      for(int j=0; j<6; ++j)
                        {
                          float xx = mV[j][i1];
                          float zz = mV[j][i];
                          mV[j][i1] = xx*c + zz*s;
                          mV[j][i] = -xx*s + zz*c;
                        }
                    z = sqrt(f*f + h*h);
                    vDiagonal[i1] = z;
                    if(z!=0)
                      {
                        c = f/z;
                        s = h/z;
                      }
                    f = c*g + s*y;
                    x = -s*g + c*y;
                    if(1)
                      for(int j=0; j<6; ++j)
                        {
                          float yy = mU[j][i1];
                          float zz = mU[j][i];
                          mU[j][i1] = yy*c + zz*s;
                          mU[j][i] = -yy*s + zz*c;
                        }
		  } // 600
		vOffDiagonal[l] = 0;
		vOffDiagonal[k] = f;
		vDiagonal[k] = x;
		result = 0; // convergence.
		break;
		// EO IF NOT CONVERGED CHUNK
	      } // EO IF TESTS HOLD
	    } // 530
	  // Code should never get here!

	  bConverged_Or_Error = result;
	}


      if(nError) {
	anticipate_exit = 1;
	break;
      }

      if(z < 0)
	{
	  vDiagonal[k] = -z;
	  if(1)
	    for(int j=0; j<6; ++j)
	      mV[j][k] = -mV[j][k];
	}
    } // 700


  if (anticipate_exit) {

  } else {
  

  float inv_diag[6];

  float dMax = vDiagonal[0];
  for(int i=1; i<6; ++i) dMax = max(dMax, vDiagonal[i]);

  for(int i=0; i<6; ++i)
    inv_diag[i] = (vDiagonal[i] * 1e6 > dMax) ? 1/vDiagonal[i] : 0;


  float b[6];
  b[0] = output[1];
  b[1] = output[2];
  b[2] = output[3];
  b[3] = output[4];
  b[4] = output[5];
  b[5] = output[6];

  // Transpose mU
  float TmU[6][6];
  for(int i=0; i<6; ++i)
    for(int j=0; j<6; ++j) {
      TmU[j][i] = mU[i][j] ;
    }




  float vTmUfvb[6];// = vTmU * vb;

  for(int i=0; i<6; i++) {
    vTmUfvb[i] = 0;
    for(int k = 0; k<6; k++)
      vTmUfvb[i] += TmU[i][k]*b[k];
  }

  float diagmultres [6];//= diagmult(vinv_diag, vTmUfvb) ;

  for(int i=0; i<6; i++) {
    diagmultres[i]  = inv_diag[i]*vTmUfvb[i];
  }


  float x [6];

  for(int i=0; i<6; i++) {
    x[i] = 0;
    for(int k = 0; k<6; k++)
      x[i] += mV[i][k]*diagmultres[k];
  }


  // From here only MatMult, Ok

  const float one_6th = 1.0/6.0;
  const float one_20th = 1.0/20.0;


  float  my_rotation_matrix[3][3];
  float my_translation[3];


  float w[3];
  w[0] = x[3];
  w[1] = x[4];
  w[2] = x[5];
  float xf[3];
  xf[0] = x[0];
  xf[1] = x[1];
  xf[2] = x[2];
  const float theta_sq = w[0]*w[0] + w[1]*w[1] + w[2]*w[2] ;
  const float theta = sqrt(theta_sq);
  float A, B;

  float cross[3];

  cross[0] = w[1]*xf[2] - w[2]*xf[1];
  cross[1] = w[2]*xf[0] - w[0]*xf[2];
  cross[2] = w[0]*xf[1] - w[1]*xf[0];

  if (theta_sq < 1e-8) {
    A = 1.0 - one_6th * theta_sq;
    B = 0.5;
    my_translation[0] = xf[0] + 0.5 * cross[0];
    my_translation[1] = xf[1] + 0.5 * cross[1];
    my_translation[2] = xf[2] + 0.5 * cross[2];
  } else {
    float C;
    if (theta_sq < 1e-6) {
      C = one_6th*(1.0 - one_20th * theta_sq);
      A = 1.0 - theta_sq * C;
      B = 0.5 - 0.25 * one_6th * theta_sq;
    } else {
      const float inv_theta = 1.0/theta;
      A = sin(theta) * inv_theta;
      B = (1 - cos(theta)) * (inv_theta * inv_theta);
      C = (1 - A) * (inv_theta * inv_theta);
    }

    float wcross[3];

    wcross[0] = w[1]*cross[2] - w[2]*cross[1];
    wcross[1] = w[2]*cross[0] - w[0]*cross[2];
    wcross[2] = w[0]*cross[1] - w[1]*cross[0];

    float Bcross[3];
    Bcross[0] = B *  cross[0];
    Bcross[1] = B *  cross[1];
    Bcross[2] = B *  cross[2];

    float Ccross[3];
    Ccross[0] = C *  cross[0];
    Ccross[1] = C *  cross[1];
    Ccross[2] = C *  cross[2];


    my_translation[0] = xf[0] + Bcross[0] + Ccross[0];
    my_translation[1] = xf[1] + Bcross[1] + Ccross[1];
    my_translation[2] = xf[2] + Bcross[2] + Ccross[2];
  }

  //rodrigues_so3_exp(w, A, B, result.get_rotation().my_matrix);
  {
    const float wx2 = (float)w[0]*w[0];
    const float wy2 = (float)w[1]*w[1];
    const float wz2 = (float)w[2]*w[2];

    my_rotation_matrix[0][0] = 1.0 - B*(wy2 + wz2);
    my_rotation_matrix[1][1] = 1.0 - B*(wx2 + wz2);
    my_rotation_matrix[2][2] = 1.0 - B*(wx2 + wy2);
  }
  {
    const float a = A*w[2];
    const float b = B*(w[0]*w[1]);
    my_rotation_matrix[0][1] = b - a;
    my_rotation_matrix[1][0] = b + a;
  }
  {
    const float a = A*w[1];
    const float b = B*(w[0]*w[2]);
    my_rotation_matrix[0][2] = b + a;
    my_rotation_matrix[2][0] = b - a;
  }
  {
    const float a = A*w[0];
    const float b = B*(w[1]*w[2]);
    my_rotation_matrix[1][2] = b - a;
    my_rotation_matrix[2][1] = b + a;
  }

  /*
    my_rotation_matrix[0] = myunit(my_rotation_matrix[0]);
    my_rotation_matrix[1] -= my_rotation_matrix[0] * (my_rotation_matrix[0]*my_rotation_matrix[1]);
    my_rotation_matrix[1] = myunit(my_rotation_matrix[1]);
    my_rotation_matrix[2] -= my_rotation_matrix[0] * (my_rotation_matrix[0]*my_rotation_matrix[2]);
    my_rotation_matrix[2] -= my_rotation_matrix[1] * (my_rotation_matrix[1]*my_rotation_matrix[2]);
    my_rotation_matrix[2] = myunit(my_rotation_matrix[2]);
  */

  {
    float vv = my_rotation_matrix[0][0] *  my_rotation_matrix[0][0] +my_rotation_matrix[0][1] *  my_rotation_matrix[0][1] +my_rotation_matrix[0][2] *  my_rotation_matrix[0][2] ;
    float coef = (1/sqrt(vv));
    my_rotation_matrix[0][0] = my_rotation_matrix[0][0] * coef;
    my_rotation_matrix[0][1] = my_rotation_matrix[0][1] * coef;
    my_rotation_matrix[0][2] = my_rotation_matrix[0][2] * coef;
  }

  {

    float my_rotation_matrix01 =  my_rotation_matrix[0][0] *  my_rotation_matrix[1][0]
      +my_rotation_matrix[0][1] *  my_rotation_matrix[1][1]
      +my_rotation_matrix[0][2] *  my_rotation_matrix[1][2] ;

    float my_rotation_matrix001[3];
    my_rotation_matrix001[0] = my_rotation_matrix[0][0] * my_rotation_matrix01;
    my_rotation_matrix001[1] = my_rotation_matrix[0][1] * my_rotation_matrix01;
    my_rotation_matrix001[2] = my_rotation_matrix[0][2] * my_rotation_matrix01;

    my_rotation_matrix[1][0] -= my_rotation_matrix001[0];
    my_rotation_matrix[1][1] -= my_rotation_matrix001[1];
    my_rotation_matrix[1][2] -= my_rotation_matrix001[2];

  }



  {
    float vv = my_rotation_matrix[1][0] *  my_rotation_matrix[1][0] +my_rotation_matrix[1][1] *  my_rotation_matrix[1][1] +my_rotation_matrix[1][2] *  my_rotation_matrix[1][2] ;
    float coef = (1/sqrt(vv));
    my_rotation_matrix[1][0] = my_rotation_matrix[1][0] * coef;
    my_rotation_matrix[1][1] = my_rotation_matrix[1][1] * coef;
    my_rotation_matrix[1][2] = my_rotation_matrix[1][2] * coef;
  }

  {

    float my_rotation_matrix02 =  my_rotation_matrix[0][0] *  my_rotation_matrix[2][0]
      +my_rotation_matrix[0][1] *  my_rotation_matrix[2][1]
      +my_rotation_matrix[0][2] *  my_rotation_matrix[2][2] ;

    float my_rotation_matrix002[3];
    my_rotation_matrix002[0] = my_rotation_matrix[0][0] * my_rotation_matrix02;
    my_rotation_matrix002[1] = my_rotation_matrix[0][1] * my_rotation_matrix02;
    my_rotation_matrix002[2] = my_rotation_matrix[0][2] * my_rotation_matrix02;

    my_rotation_matrix[2][0] -= my_rotation_matrix002[0];
    my_rotation_matrix[2][1] -= my_rotation_matrix002[1];
    my_rotation_matrix[2][2] -= my_rotation_matrix002[2];

  }

  {

    float my_rotation_matrix12 =  my_rotation_matrix[1][0] *  my_rotation_matrix[2][0]
      +my_rotation_matrix[1][1] *  my_rotation_matrix[2][1]
      +my_rotation_matrix[1][2] *  my_rotation_matrix[2][2] ;

    float my_rotation_matrix112[3];
    my_rotation_matrix112[0] = my_rotation_matrix[1][0] * my_rotation_matrix12;
    my_rotation_matrix112[1] = my_rotation_matrix[1][1] * my_rotation_matrix12;
    my_rotation_matrix112[2] = my_rotation_matrix[1][2] * my_rotation_matrix12;

    my_rotation_matrix[2][0] -= my_rotation_matrix112[0];
    my_rotation_matrix[2][1] -= my_rotation_matrix112[1];
    my_rotation_matrix[2][2] -= my_rotation_matrix112[2];

  }




  {
    float vv = my_rotation_matrix[2][0] *  my_rotation_matrix[2][0] +my_rotation_matrix[2][1] *  my_rotation_matrix[2][1] +my_rotation_matrix[2][2] *  my_rotation_matrix[2][2] ;
    float coef = (1/sqrt(vv));
    my_rotation_matrix[2][0] = my_rotation_matrix[2][0] * coef;
    my_rotation_matrix[2][1] = my_rotation_matrix[2][1] * coef;
    my_rotation_matrix[2][2] = my_rotation_matrix[2][2] * coef;
  }





  float I [4][4];
  for (int i = 0 ; i < 4 ; i++) {
    for (int j = 0 ; j < 4 ; j++) {
      I[i][j] = 0;
    }
  }
  for (int i = 0 ; i < 4 ; i++) {
    I[i][i] = 1;
  }


  float TmpRes[4][4];
  float Itranspose[4][4];
  for(int i=0; i<4; ++i)
    for(int j=0; j<4; ++j) {
      Itranspose[j][i] = I[i][j] ;
    }

  for(int i=0; i<4; ++i) {
    //TmpRes[i].slice<0,3>()=my_rotation_matrix *  Itranspose[i].slice<0,3>();

    TmpRes[i][0] = my_rotation_matrix[0][0] *  Itranspose[i][0]
      + my_rotation_matrix[0][1] *  Itranspose[i][1]
      + my_rotation_matrix[0][2] *  Itranspose[i][2] ;
    TmpRes[i][1] = my_rotation_matrix[1][0] *  Itranspose[i][0]
      + my_rotation_matrix[1][1] *  Itranspose[i][1]
      + my_rotation_matrix[1][2] *  Itranspose[i][2] ;
    TmpRes[i][2] = my_rotation_matrix[2][0] *  Itranspose[i][0]
      + my_rotation_matrix[2][1] *  Itranspose[i][1]
      + my_rotation_matrix[2][2] *  Itranspose[i][2] ;

    TmpRes[i][0] += my_translation[0] *  Itranspose[i][3];
    TmpRes[i][1] += my_translation[1] *  Itranspose[i][3];
    TmpRes[i][2] += my_translation[2] *  Itranspose[i][3];
    TmpRes[i][3] =  Itranspose[i][3];
  }
  Matrix4 RR;

  for(int j=0; j<4; ++j) {
    RR.data[j].x =  TmpRes[0][j] ;
    RR.data[j].y =  TmpRes[1][j] ;
    RR.data[j].z =  TmpRes[2][j] ;
    RR.data[j].w =  TmpRes[3][j] ;

  }
  Matrix4 m1 = pose;
  Matrix4 m2 = RR;
  Matrix4 m3;
  for (int i = 0; i < 4; i++ ) {
    m3.data[ i ].x = 0;
    m3.data[ i ].x += m1.data[ i ].x * m2.data[ 0 ].x;
    m3.data[ i ].x += m1.data[ i ].y * m2.data[ 1 ].x;
    m3.data[ i ].x += m1.data[ i ].z * m2.data[ 2 ].x;
    m3.data[ i ].x += m1.data[ i ].w * m2.data[ 3 ].x;
      
    m3.data[ i ].y = 0;
    m3.data[ i ].y += m1.data[ i ].x * m2.data[ 0 ].y;
    m3.data[ i ].y += m1.data[ i ].y * m2.data[ 1 ].y;
    m3.data[ i ].y += m1.data[ i ].z * m2.data[ 2 ].y;
    m3.data[ i ].y += m1.data[ i ].w * m2.data[ 3 ].y;
      
    m3.data[ i ].z = 0;
    m3.data[ i ].z += m1.data[ i ].x * m2.data[ 0 ].z;
    m3.data[ i ].z += m1.data[ i ].y * m2.data[ 1 ].z;
    m3.data[ i ].z += m1.data[ i ].z * m2.data[ 2 ].z;
    m3.data[ i ].z += m1.data[ i ].w * m2.data[ 3 ].z;
      
    m3.data[ i ].w = 0;
    m3.data[ i ].w += m1.data[ i ].x * m2.data[ 0 ].w;
    m3.data[ i ].w += m1.data[ i ].y * m2.data[ 1 ].w;
    m3.data[ i ].w += m1.data[ i ].z * m2.data[ 2 ].w;
    m3.data[ i ].w += m1.data[ i ].w * m2.data[ 3 ].w;
  }

  //pose->data[0] = m3.data[0];
  //pose->data[1] = m3.data[1];
  //pose->data[2] = m3.data[2];
  //pose->data[3] = m3.data[3];

  // Return validity test result of the tracking
  float xsqr = 0;
  for(int i=0; i<6; ++i) {
    xsqr += x[i] * x[i];
  }
  float lnorm =     sqrt(xsqr);

  /* skipped test : lnorm < icp_threshold */
  }
}


inline Matrix4 getInverseCameraMatrix(const float4  k) {
	Matrix4 invK;
	invK.data[0] = make_float4(1.0f / k.x, 0, -k.z / k.x, 0);
	invK.data[1] = make_float4(0, 1.0f / k.y, -k.w / k.y, 0);
	invK.data[2] = make_float4(0, 0, 1, 0);
	invK.data[3] = make_float4(0, 0, 0, 1);
	return invK;
}




int tracking_pencil(unsigned int size0x, unsigned int size0y,
		    unsigned int size1x, unsigned int size1y,
		    unsigned int size2x, unsigned int size2y,
		    float ScaledDepth0[restrict const static size0y][size0x],
		    float ScaledDepth1[restrict const static size1y][size1x],
		    float ScaledDepth2[restrict const static size2y][size2x],
		    float3 InputVertex0[restrict const static size0y][size0x],
		    float3 InputVertex1[restrict const static size1y][size1x],
		    float3 InputVertex2[restrict const static size2y][size2x],
		    float3 InputNormal0[restrict const static size0y][size0x],
		    float3 InputNormal1[restrict const static size1y][size1x],
		    float3 InputNormal2[restrict const static size2y][size2x],
		    float3 refVertex[restrict const static size0y][size0x],
		    float3 refNormal[restrict const static size0y][size0x],
		    TrackData trackingResult[restrict const static size0y][size0x],
		    float reductionoutput[restrict const static 8][32] ,
		    const Matrix4 pose, const Matrix4 projectReference,
		    const float dist_threshold, const float normal_threshold,		   
		    Matrix4 invK0, Matrix4 invK1, Matrix4 invK2,
		    int iterations0 ,
		    int iterations1 ,
		    int iterations2 ,
		    float e_delta		    ) {



  
  inline_halfSampleRobustImage_pencil(size1x, size1y, size0x, size0y, ScaledDepth1, ScaledDepth0, e_delta * 3, 1);
  inline_halfSampleRobustImage_pencil(size2x, size2y, size1x, size1y, ScaledDepth2, ScaledDepth1, e_delta * 3, 1);

  inline_depth2vertex_pencil(size0x, size0y, InputVertex0, ScaledDepth0, invK0);
  inline_vertex2normal_pencil(size0x, size0y, InputNormal0, InputVertex0);

  inline_depth2vertex_pencil(size1x, size1y, InputVertex1, ScaledDepth1, invK1);
  inline_vertex2normal_pencil(size1x, size1y, InputNormal1, InputVertex1);

  inline_depth2vertex_pencil(size2x, size2y, InputVertex2, ScaledDepth2, invK2);
  inline_vertex2normal_pencil(size2x, size2y, InputNormal2, InputVertex2);

  for (int i = 0; i < iterations2; ++i) {

    inline_track_pencil(size0x, size0y,  size2x, size2y,  trackingResult,  InputVertex2, InputNormal2, refVertex, refNormal,  pose,  projectReference,  dist_threshold,  normal_threshold);

    inline_reduce_pencil(reductionoutput, size0x, size0y,  trackingResult, size2x, size2y);
    inline_update_pose_pencil(pose, reductionoutput[0]);
    //inline_original_update_pose_pencil(pose, reductionoutput[0]);
  }

  for (int i = 0; i < iterations1; ++i) {

    inline_track_pencil(size0x, size0y,  size1x, size1y,  trackingResult,  InputVertex1, InputNormal1, refVertex, refNormal,  pose,  projectReference,  dist_threshold,  normal_threshold);

    inline_reduce_pencil(reductionoutput, size0x, size0y,  trackingResult, size1x, size1y);
    inline_update_pose_pencil(pose, reductionoutput[0]);
    //inline_original_update_pose_pencil(pose, reductionoutput[0]);
  }
  
  for (int i = 0; i < iterations0; ++i) {

    inline_track_pencil(size0x, size0y,  size0x, size0y,  trackingResult,  InputVertex0, InputNormal0, refVertex, refNormal,  pose,  projectReference,  dist_threshold,  normal_threshold);

    inline_reduce_pencil(reductionoutput, size0x, size0y,  trackingResult, size0x, size0y);
    inline_update_pose_pencil(pose, reductionoutput[0]);
    //inline_original_update_pose_pencil(pose, reductionoutput[0]);
  }


  
  return 0;
  
}

int preprocessing_pencil(
			 const uint inputSizex,
			 const uint inputSizey,
			 const unsigned short inputDepth[restrict const static inputSizey][inputSizex],
			 const uint computationSizex,
			 const uint computationSizey,
			 float    floatDepth[restrict const static computationSizey][computationSizex],
			 float    ScaledDepth[restrict const static computationSizey][computationSizex],
			 int radius,
			 float* gaussian,
			 float e_delta)
{

	int ratio = inputSizex / computationSizex;
	uint2 computationSize = {computationSizex,computationSizey};

	#pragma scop
	
	inline_mm2meters_pencil(computationSizex, computationSizey, floatDepth, inputSizex, inputSizey, inputDepth, ratio);

	
	inline_bilateralFilter_pencil(computationSizex, computationSizey, ScaledDepth , floatDepth, computationSize, (radius * 2 + 1), gaussian, e_delta, radius);

	#pragma endscop
	
	return 0;
}
