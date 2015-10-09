#include <kernels.h>
#include <stdio.h>
#include <string.h>

#define PENCIL_LIB_H
//defining PENCIL_LIB_H is needed to avoid function definition conflicts between
//pencil_lib.h and cutil_math.h. pencil_lib.h is included from within prl.h.
#include <prl.h>

#define TICK()    {if (print_kernel_timing) {clock_gettime(CLOCK_MONOTONIC, &tick_clockData);}
#define TOCK(str,size)  if (print_kernel_timing) {clock_gettime(CLOCK_MONOTONIC, &tock_clockData); std::cerr<< str << " ";\
	if((tock_clockData.tv_sec > tick_clockData.tv_sec) && (tock_clockData.tv_nsec >= tick_clockData.tv_nsec))   std::cerr<< tock_clockData.tv_sec - tick_clockData.tv_sec << std::setfill('0') << std::setw(9);\
	std::cerr  << (( tock_clockData.tv_nsec - tick_clockData.tv_nsec) + ((tock_clockData.tv_nsec<tick_clockData.tv_nsec)?1000000000:0)) << " " <<  size << std::endl;}}

extern "C" {
	int bilateralFilter_pencil(int size_x, int size_y, float *out, const float *in, uint2 size, int gaussianS, const float *gaussian, float e_d, int r);

	int mm2meters_pencil(uint outSize_x, uint outSize_y, float *out, uint inSize_x, uint inSize_y, const ushort *in, int ratio);

	int initVolume_pencil(const uint v_size_x, const uint v_size_y, const uint v_size_z, short2 *v_data, const float2 d);

	int integrateKernel_pencil(const uint vol_size_x, const uint vol_size_y, const uint vol_size_z, const float3 vol_dim, short2 *vol_data, uint depthSize_x,
	                           uint depthSize_y, const float *depth, const Matrix4 invTrack, const Matrix4 K, const float mu, const float maxweight);

	int depth2vertex_pencil(uint imageSize_x, uint imageSize_y, float3 *vertex, const float *depth, const Matrix4 invK);

	int vertex2normal_pencil(uint imageSize_x, uint imageSize_y, float3 *out, const float3 *in);

	int track_pencil(uint refSize_x, uint refSize_y, uint inSize_x, uint inSize_y, TrackData *output, const float3 *inVertex,
	                 const float3 *inNormal, const float3 *refVertex, const float3 *refNormal, const Matrix4 Ttrack,
	                 const Matrix4 view, const float dist_threshold,	const float normal_threshold);

	int halfSampleRobustImage_pencil(uint outSize_x, uint outSize_y, uint inSize_x, uint inSize_y, float *out, const float *in, const float e_d, const int r);

	int renderNormal_pencil(uint normalSize_x, uint normalSize_y, uchar3 *out, const float3 *normal);

	int renderDepth_pencil(uint depthSize_x, uint depthSize_y, uchar4 *out, const float *depth, const float nearPlane, const float farPlane);

	int renderTrack_pencil(uint outSize_x, uint outSize_y, uchar4 *out, const TrackData *data);

	int renderVolume_pencil(uint depthSize_x, uint depthSize_y, uchar4 *out, const uint volume_size_x, const uint volume_size_y, const uint volume_size_z,
	                        const short2 *volume_data, const float3 volume_dim, const Matrix4 view, const float nearPlane, const float farPlane,
	                        const float step, const float largestep, const float3 light, const float3 ambient);

	int raycast_pencil(uint inputSize_x, uint inputSize_y, float3 *vertex, float3 *normal, const uint integration_size_x, const uint integration_size_y,
	                   const uint integration_size_z, const short2 *integration_data, const float3 integration_dim, const Matrix4 view,
	                   const float nearPlane, const float farPlane, const float step, const float largestep);

	int reduce_pencil(float *sums, const uint Jsize_x, const uint Jsize_y, TrackData *J, const uint size_x, const uint size_y);
}

float * gaussian;

Volume volume;
float3 *vertex;
float3 *normal;

TrackData *trackingResult;
float *reductionoutput;
float **ScaledDepth;
float *floatDepth;
Matrix4 oldPose;
Matrix4 raycastPose;
float3 **inputVertex;
float3 **inputNormal;

bool print_kernel_timing = false;
struct timespec tick_clockData;
struct timespec tock_clockData;

void Kfusion::languageSpecificConstructor()
{
	if (getenv("KERNEL_TIMINGS"))
		print_kernel_timing = true;

	prl_init();

	// internal buffers to initialize
	size_t reductionoutput_size = sizeof(float) * 8 * 32;
	reductionoutput = (float*) prl_alloc(reductionoutput_size);
	memset(reductionoutput, 0, reductionoutput_size);

	ScaledDepth = (float**)  malloc(sizeof(float*)  * iterations.size());
	inputVertex = (float3**) malloc(sizeof(float3*) * iterations.size());
	inputNormal = (float3**) malloc(sizeof(float3*) * iterations.size());

	for (unsigned int i = 0; i < iterations.size(); ++i) {
		size_t size = (computationSize.x * computationSize.y) / (int) pow(2, i);
		ScaledDepth[i] = (float*)  prl_alloc(sizeof(float)  * size);
		memset(ScaledDepth[i], 0, sizeof(float) * size);

		inputVertex[i] = (float3*) prl_alloc(sizeof(float3) * size);
		memset(inputVertex[i], 0, sizeof(float3) * size);

		inputNormal[i] = (float3*) prl_alloc(sizeof(float3) * size);
		memset(inputNormal[i], 0, sizeof(float3) * size);
	}

	size_t size = computationSize.x * computationSize.y;
	floatDepth     = (float*)     prl_alloc(sizeof(float)     * size);
	vertex         = (float3*)    prl_alloc(sizeof(float3)    * size);
	normal         = (float3*)    prl_alloc(sizeof(float3)    * size);
	trackingResult = (TrackData*) prl_alloc(sizeof(TrackData) * size);

	memset(floatDepth, 0, sizeof(float) * size);
	memset(vertex, 0, sizeof(float3) * size);
	memset(normal, 0, sizeof(float3) * size);
	memset(trackingResult, 0, sizeof(TrackData) * size);

	// Start generating the gaussian.
	size_t gaussianS = radius * 2 + 1;
	gaussian = (float*) prl_alloc(gaussianS * sizeof(float));
	int x;
	for (unsigned int i = 0; i < gaussianS; i++) {
		x = i - 2;
		gaussian[i] = expf(-(x * x) / (2 * delta * delta));
	}
	// Done generating the gaussian.

	volume.init(volumeResolution, volumeDimensions);
	reset();
}

Kfusion::~Kfusion()
{
	prl_free(reductionoutput);
	for (unsigned int i = 0; i < iterations.size(); ++i) {
		prl_free(ScaledDepth[i]);
		prl_free(inputVertex[i]);
		prl_free(inputNormal[i]);
	}
	free(ScaledDepth);
	free(inputVertex);
	free(inputNormal);

	prl_free(vertex);
	prl_free(normal);
	prl_free(gaussian);
	prl_free(floatDepth);
	prl_free(trackingResult);

	//prl_shutdown();

	volume.release();
}

void Kfusion::reset()
{
	initVolumeKernel(volume);
}

void init() {};

void clean() {};

void initVolumeKernel(Volume volume)
{
	TICK();

//	for (unsigned int x = 0; x < volume.size.x; x++)
//		for (unsigned int y = 0; y < volume.size.y; y++) {
//			for (unsigned int z = 0; z < volume.size.z; z++) {
//				//std::cout <<  x << " " << y << " " << z <<"\n";
//				volume.setints(x, y, z, make_float2(1.0f, 0.0f));
//			}
//		}

	initVolume_pencil(volume.size.x, volume.size.y, volume.size.z,
	                  volume.data, make_float2(1.0f, 0.0f));
	TOCK("initVolumeKernel", volume.size.x * volume.size.y * volume.size.z);
}

void bilateralFilterKernel(float *out, const float *in, uint2 size,
                           const float *gaussian, float e_d, int r)
{
	TICK();

//	uint y;
//	float e_d_squared_2 = e_d * e_d * 2;
//#pragma omp parallel for \
//    shared(out),private(y)
//	for (y = 0; y < size.y; y++) {
//		for (uint x = 0; x < size.x; x++) {
//			uint pos = x + y * size.x;
//			if (in[pos] == 0) {
//				out[pos] = 0;
//				continue;
//			}
//
//			float sum = 0.0f;
//			float t = 0.0f;
//
//			const float center = in[pos];
//
//			for (int i = -r; i <= r; ++i) {
//				for (int j = -r; j <= r; ++j) {
//					uint2 curPos = make_uint2(clamp(x + i, 0u, size.x - 1),
//							clamp(y + j, 0u, size.y - 1));
//					const float curPix = in[curPos.x + curPos.y * size.x];
//					if (curPix > 0) {
//						const float mod = sq(curPix - center);
//						const float factor = gaussian[i + r]
//								* gaussian[j + r]
//								* expf(-mod / e_d_squared_2);
//						t += factor * curPix;
//						sum += factor;
//					}
//				}
//			}
//			out[pos] = t / sum;
//		}
//	}

	bilateralFilter_pencil(size.x, size.y, out, in, size,
	                       (radius * 2 + 1), gaussian, e_d, r);
	TOCK("bilateralFilterKernel", size.x * size.y);
}

void depth2vertexKernel(float3 *vertex, const float *depth,
                        uint2 imageSize, const Matrix4 invK)
{
	TICK();

//	unsigned int x, y;
//#pragma omp parallel for \
//         shared(vertex), private(x, y)
//	for (y = 0; y < imageSize.y; y++) {
//		for (x = 0; x < imageSize.x; x++) {
//
//			if (depth[x + y * imageSize.x] > 0) {
//				vertex[x + y * imageSize.x] = depth[x + y * imageSize.x]
//						* (rotate(invK, make_float3(x, y, 1.f)));
//			} else {
//				vertex[x + y * imageSize.x] = make_float3(0);
//			}
//		}
//	}
	depth2vertex_pencil(imageSize.x, imageSize.y, vertex, depth, invK);
	TOCK("depth2vertexKernel", imageSize.x * imageSize.y);
}

void vertex2normalKernel(float3 *out, const float3 *in, uint2 imageSize)
{
	TICK();

//	unsigned int x, y;
//#pragma omp parallel for \
//        shared(out), private(x,y)
//	for (y = 0; y < imageSize.y; y++) {
//		for (x = 0; x < imageSize.x; x++) {
//			const uint2 pleft = make_uint2(max(int(x) - 1, 0), y);
//			const uint2 pright = make_uint2(min(x + 1, (int) imageSize.x - 1),
//					y);
//			const uint2 pup = make_uint2(x, max(int(y) - 1, 0));
//			const uint2 pdown = make_uint2(x,
//					min(y + 1, ((int) imageSize.y) - 1));
//
//			const float3 left = in[pleft.x + imageSize.x * pleft.y];
//			const float3 right = in[pright.x + imageSize.x * pright.y];
//			const float3 up = in[pup.x + imageSize.x * pup.y];
//			const float3 down = in[pdown.x + imageSize.x * pdown.y];
//
//			if (left.z == 0 || right.z == 0 || up.z == 0 || down.z == 0) {
//				out[x + y * imageSize.x].x = INVALID;
//				continue;
//			}
//			const float3 dxv = right - left;
//			const float3 dyv = down - up;
//			out[x + y * imageSize.x] = normalize(cross(dyv, dxv)); // switched dx and dy to get factor -1
//		}
//	}

	vertex2normal_pencil(imageSize.x, imageSize.y, out, in);
	TOCK("vertex2normalKernel", imageSize.x * imageSize.y);
}

void trackKernel(TrackData* output, const float3* inVertex,
                 const float3* inNormal, uint2 inSize, const float3* refVertex,
                 const float3* refNormal, uint2 refSize, const Matrix4 Ttrack,
                 const Matrix4 view, const float dist_threshold,
                 const float normal_threshold)
{
	TICK();

	uint2 pixel = make_uint2(0, 0);
//	unsigned int pixely, pixelx;
//#pragma omp parallel for \
//	    shared(output), private(pixel,pixelx,pixely)
//	for (pixely = 0; pixely < inSize.y; pixely++) {
//		for (pixelx = 0; pixelx < inSize.x; pixelx++) {
//			pixel.x = pixelx;
//			pixel.y = pixely;
//
//			TrackData & row = output[pixel.x + pixel.y * refSize.x];
//
//			if (inNormal[pixel.x + pixel.y * inSize.x].x == INVALID) {
//				row.result = -1;
//				continue;
//			}
//
//			const float3 projectedVertex = Ttrack
//					* inVertex[pixel.x + pixel.y * inSize.x];
//			const float3 projectedPos = view * projectedVertex;
//			const float2 projPixel = make_float2(
//					projectedPos.x / projectedPos.z + 0.5f,
//					projectedPos.y / projectedPos.z + 0.5f);
//			if (projPixel.x < 0 || projPixel.x > refSize.x - 1
//					|| projPixel.y < 0 || projPixel.y > refSize.y - 1) {
//				row.result = -2;
//				continue;
//			}
//
//			const uint2 refPixel = make_uint2(projPixel.x, projPixel.y);
//			const float3 referenceNormal = refNormal[refPixel.x
//					+ refPixel.y * refSize.x];
//
//			if (referenceNormal.x == INVALID) {
//				row.result = -3;
//				continue;
//			}
//
//			const float3 diff = refVertex[refPixel.x + refPixel.y * refSize.x]
//					- projectedVertex;
//			const float3 projectedNormal = rotate(Ttrack,
//					inNormal[pixel.x + pixel.y * inSize.x]);
//
//			if (length(diff) > dist_threshold) {
//				row.result = -4;
//				continue;
//			}
//			if (dot(projectedNormal, referenceNormal) < normal_threshold) {
//				row.result = -5;
//				continue;
//			}
//			row.result = 1;
//			row.error = dot(referenceNormal, diff);
//			((float3 *) row.J)[0] = referenceNormal;
//			((float3 *) row.J)[1] = cross(projectedVertex, referenceNormal);
//		}
//	}
	track_pencil(refSize.x, refSize.y, inSize.x, inSize.y, output,
	             inVertex, inNormal, refVertex, refNormal, Ttrack,
	             view, dist_threshold, normal_threshold);
	TOCK("trackKernel", inSize.x * inSize.y);
}

void mm2metersKernel(float * out, uint2 outSize,
                     const ushort * in, uint2 inSize)
{
	TICK();
	// Check for unsupported conditions
	if ((inSize.x < outSize.x) || (inSize.y < outSize.y)) {
		std::cerr << "Invalid ratio." << std::endl;
		exit(1);
	}
	if ((inSize.x % outSize.x != 0) || (inSize.y % outSize.y != 0)) {
		std::cerr << "Invalid ratio." << std::endl;
		exit(1);
	}
	if ((inSize.x / outSize.x != inSize.y / outSize.y)) {
		std::cerr << "Invalid ratio." << std::endl;
		exit(1);
	}

	int ratio = inSize.x / outSize.x;
//	unsigned int y;
//#pragma omp parallel for \
//        shared(out), private(y)
//	for (y = 0; y < outSize.y; y++)
//		for (unsigned int x = 0; x < outSize.x; x++) {
//			out[x + outSize.x * y] = in[x * ratio + inSize.x * y * ratio]
//					/ 1000.0f;
//		}

	mm2meters_pencil(outSize.x, outSize.y, out, inSize.x, inSize.y, in, ratio);
	TOCK("mm2metersKernel", outSize.x * outSize.y);
}

void halfSampleRobustImageKernel(float* out, const float* in, uint2 inSize,
                                 const float e_d, const int r)
{
	TICK();

//	uint2 outSize = make_uint2(inSize.x / 2, inSize.y / 2);
//	unsigned int y;
//#pragma omp parallel for \
//        shared(out), private(y)
//	for (y = 0; y < outSize.y; y++) {
//		for (unsigned int x = 0; x < outSize.x; x++) {
//			uint2 pixel = make_uint2(x, y);
//			const uint2 centerPixel = 2 * pixel;
//
//			float sum = 0.0f;
//			float t = 0.0f;
//			const float center = in[centerPixel.x
//					+ centerPixel.y * inSize.x];
//			for (int i = -r + 1; i <= r; ++i) {
//				for (int j = -r + 1; j <= r; ++j) {
//					uint2 cur = make_uint2(
//							clamp(
//									make_int2(centerPixel.x + j,
//											centerPixel.y + i), make_int2(0),
//									make_int2(2 * outSize.x - 1,
//											2 * outSize.y - 1)));
//					float current = in[cur.x + cur.y * inSize.x];
//					if (fabsf(current - center) < e_d) {
//						sum += 1.0f;
//						t += current;
//					}
//				}
//			}
//			out[pixel.x + pixel.y * outSize.x] = t / sum;
//		}
//	}
        //TODO: check the outsize param
	halfSampleRobustImage_pencil(inSize.x, inSize.y, 2*inSize.x, 2*inSize.y,
	                             out, in, e_d, r);
	TOCK("halfSampleRobustImageKernel", inSize.x * inSize.y);
}

void integrateKernel(Volume vol, const float* depth, uint2 depthSize,
                     const Matrix4 invTrack, const Matrix4 K, const float mu,
                     const float maxweight)
{
	TICK();

//	const float3 delta = rotate(invTrack,
//			make_float3(0, 0, vol.dim.z / vol.size.z));
//	const float3 cameraDelta = rotate(K, delta);
//	unsigned int y;
//#pragma omp parallel for \
//        shared(vol), private(y)
//	for (y = 0; y < vol.size.y; y++)
//		for (unsigned int x = 0; x < vol.size.x; x++) {
//
//			uint3 pix = make_uint3(x, y, 0); //pix.x = x;pix.y = y;
//			float3 pos = invTrack * vol.pos(pix);
//			float3 cameraX = K * pos;
//
//			for (pix.z = 0; pix.z < vol.size.z;
//					++pix.z, pos += delta, cameraX += cameraDelta) {
//				if (pos.z < 0.0001f) // some near plane constraint
//					continue;
//				const float2 pixel = make_float2(cameraX.x / cameraX.z + 0.5f,
//						cameraX.y / cameraX.z + 0.5f);
//				if (pixel.x < 0 || pixel.x > depthSize.x - 1 || pixel.y < 0
//						|| pixel.y > depthSize.y - 1)
//					continue;
//				const uint2 px = make_uint2(pixel.x, pixel.y);
//				if (depth[px.x + px.y * depthSize.x] == 0)
//					continue;
//				const float diff =
//						(depth[px.x + px.y * depthSize.x] - cameraX.z)
//								* std::sqrt(
//										1 + sq(pos.x / pos.z)
//												+ sq(pos.y / pos.z));
//				if (diff > -mu) {
//					const float sdf = fminf(1.f, diff / mu);
//					float2 data = vol[pix];
//					data.x = clamp((data.y * data.x + sdf) / (data.y + 1), -1.f,
//							1.f);
//					data.y = fminf(data.y + 1, maxweight);
//					vol.set(pix, data);
//				}
//			}
//		}
	integrateKernel_pencil(vol.size.x, vol.size.y, vol.size.z, vol.dim,
	                       vol.data, depthSize.x, depthSize.y, depth,
	                       invTrack, K, mu, maxweight);
	TOCK("integrateKernel", vol.size.x * vol.size.y);
}


float4 raycast(const Volume volume, const uint2 pos, const Matrix4 view,
		const float nearPlane, const float farPlane, const float step,
		const float largestep) {

	const float3 origin = get_translation(view);
	const float3 direction = rotate(view, make_float3(pos.x, pos.y, 1.f));

	// intersect ray with a box
	// http://www.siggraph.org/education/materials/HyperGraph/raytrace/rtinter3.htm
	// compute intersection of ray with all six bbox planes
	const float3 invR = make_float3(1.0f) / direction;
	const float3 tbot = -1 * invR * origin;
	const float3 ttop = invR * (volume.dim - origin);

	// re-order intersections to find smallest and largest on each axis
	const float3 tmin = fminf(ttop, tbot);
	const float3 tmax = fmaxf(ttop, tbot);

	// find the largest tmin and the smallest tmax
	const float largest_tmin = fmaxf(fmaxf(tmin.x, tmin.y),
			fmaxf(tmin.x, tmin.z));
	const float smallest_tmax = fminf(fminf(tmax.x, tmax.y),
			fminf(tmax.x, tmax.z));

	// check against near and far plane
	const float tnear = fmaxf(largest_tmin, nearPlane);
	const float tfar = fminf(smallest_tmax, farPlane);

	if (tnear < tfar) {
		// first walk with largesteps until we found a hit
		float t = tnear;
		float stepsize = largestep;
		float f_t = volume.interp(origin + direction * t);
		float f_tt = 0;
		if (f_t > 0) { // ups, if we were already in it, then don't render anything here
			for (; t < tfar; t += stepsize) {
				f_tt = volume.interp(origin + direction * t);
				if (f_tt < 0)                  // got it, jump out of inner loop
					break;
				if (f_tt < 0.8f)               // coming closer, reduce stepsize
					stepsize = step;
				f_t = f_tt;
			}
			if (f_tt < 0) {           // got it, calculate accurate intersection
				t = t + stepsize * f_tt / (f_t - f_tt);
				return make_float4(origin + direction * t, t);
			}
		}
	}
	return make_float4(0);

}
void raycastKernel(float3* vertex, float3* normal, uint2 inputSize,
                   const Volume integration, const Matrix4 view,
                   const float nearPlane, const float farPlane,
                   const float step, const float largestep)
{
	TICK();

	unsigned int y;
#pragma omp parallel for \
	    shared(normal, vertex), private(y)
	for (y = 0; y < inputSize.y; y++)
		for (unsigned int x = 0; x < inputSize.x; x++) {

			uint2 pos = make_uint2(x, y);

			const float4 hit = raycast(integration, pos, view, nearPlane,
					farPlane, step, largestep);
			if (hit.w > 0.0) {
				vertex[pos.x + pos.y * inputSize.x] = make_float3(hit);
				float3 surfNorm = integration.grad(make_float3(hit));
				if (length(surfNorm) == 0) {
					//normal[pos] = normalize(surfNorm); // APN added
					normal[pos.x + pos.y * inputSize.x].x = INVALID;
				} else {
					normal[pos.x + pos.y * inputSize.x] = normalize(surfNorm);
				}
			} else {
				//std::cerr<< "RAYCAST MISS "<<  pos.x << " " << pos.y <<"  " << hit.w <<"\n";
				vertex[pos.x + pos.y * inputSize.x] = make_float3(0);
				normal[pos.x + pos.y * inputSize.x] = make_float3(INVALID, 0,
						0);
			}
		}

//	raycast_pencil(inputSize.x, inputSize.y, vertex, normal, integration.size.x,
//	               integration.size.y, integration.size.z, integration.data,
//	               integration.dim, view, nearPlane, farPlane, step, largestep);

	TOCK("raycastKernel", inputSize.x * inputSize.y);
}

bool updatePoseKernel(Matrix4 & pose, const float * output, float icp_threshold)
{
	bool res = false;
	TICK();
	TooN::Matrix<8, 32, const float, TooN::Reference::RowMajor> values(output);
	TooN::Vector<6> x = solve(values[0].slice<1, 27>());
	TooN::SE3<> delta(x);
	pose = toMatrix4(delta) * pose;

	if (norm(x) < icp_threshold)
		res = true;

	TOCK("updatePoseKernel", 1);
	return res;
}

bool checkPoseKernel(Matrix4 & pose, Matrix4 oldPose, const float * output,
                     uint2 imageSize, float track_threshold)
{
	TooN::Matrix<8, 32, const float, TooN::Reference::RowMajor> values(output);

	if ((std::sqrt(values(0, 0) / values(0, 28)) > 2e-2) ||
	    (values(0, 28) / (imageSize.x * imageSize.y) < track_threshold)) {
		pose = oldPose;
		return false;
	} else {
		return true;
	}

}

void renderNormalKernel(uchar3* out, const float3* normal, uint2 normalSize)
{
	TICK();


//        unsigned int y;
//#pragma omp parallel for \
//        shared(out), private(y)
//	for (y = 0; y < normalSize.y; y++)
//		for (unsigned int x = 0; x < normalSize.x; x++) {
//			uint pos = (x + y * normalSize.x);
//			float3 n = normal[pos];
//			if (n.x == -2) {
//				out[pos] = make_uchar3(0, 0, 0);
//			} else {
//				n = normalize(n);
//				out[pos] = make_uchar3(n.x * 128 + 128, n.y * 128 + 128,
//						n.z * 128 + 128);
//			}
//		}
	renderNormal_pencil(normalSize.x, normalSize.y, out, normal);
	TOCK("renderNormalKernel", normalSize.x * normalSize.y);
}

void renderDepthKernel(uchar4* out, float * depth, uint2 depthSize,
                       const float nearPlane, const float farPlane)
{
	TICK();


	float rangeScale = 1 / (farPlane - nearPlane);
//
//	unsigned int y;
//#pragma omp parallel for \
//        shared(out), private(y)
//	for (y = 0; y < depthSize.y; y++) {
//		int rowOffeset = y * depthSize.x;
//		for (unsigned int x = 0; x < depthSize.x; x++) {
//
//			unsigned int pos = rowOffeset + x;
//
//			if (depth[pos] < nearPlane)
//				out[pos] = make_uchar4(255, 255, 255, 0); // The forth value is a padding in order to align memory
//			else {
//				if (depth[pos] > farPlane)
//					out[pos] = make_uchar4(0, 0, 0, 0); // The forth value is a padding in order to align memory
//				else {
//					const float d = (depth[pos] - nearPlane) * rangeScale;
//					out[pos] = gs2rgb(d);
//				}
//			}
//		}
//	}
	renderDepth_pencil(depthSize.x, depthSize.y, out, depth, nearPlane, farPlane);
	TOCK("renderDepthKernel", depthSize.x * depthSize.y);
}

void renderTrackKernel(uchar4* out, const TrackData* data, uint2 outSize)
{
	TICK();


//	unsigned int y;
//#pragma omp parallel for \
//        shared(out), private(y)
//	for (y = 0; y < outSize.y; y++)
//		for (unsigned int x = 0; x < outSize.x; x++) {
//			uint pos = x + y * outSize.x;
//			switch (data[pos].result) {
//			case 1:
//				out[pos] = make_uchar4(128, 128, 128, 0);  // ok	 GREY
//				break;
//			case -1:
//				out[pos] = make_uchar4(0, 0, 0, 0);      // no input BLACK
//				break;
//			case -2:
//				out[pos] = make_uchar4(255, 0, 0, 0);        // not in image RED
//				break;
//			case -3:
//				out[pos] = make_uchar4(0, 255, 0, 0);    // no correspondence GREEN
//				break;
//			case -4:
//				out[pos] = make_uchar4(0, 0, 255, 0);        // to far away BLUE
//				break;
//			case -5:
//				out[pos] = make_uchar4(255, 255, 0, 0);     // wrong normal YELLOW
//				break;
//			default:
//				out[pos] = make_uchar4(255, 128, 128, 0);
//				break;
//			}
//		}

	renderTrack_pencil(outSize.x, outSize.y, out, data);
	TOCK("renderTrackKernel", outSize.x * outSize.y);
}

void renderVolumeKernel(uchar4* out, const uint2 depthSize,
                        const Volume volume, const Matrix4 view,
                        const float nearPlane, const float farPlane,
                        const float step, const float largestep,
                        const float3 light, const float3 ambient)
{
	TICK();
//	unsigned int y;
//#pragma omp parallel for \
//        shared(out), private(y)
//	for (y = 0; y < depthSize.y; y++) {
//		for (unsigned int x = 0; x < depthSize.x; x++) {
//			const uint pos = x + y * depthSize.x;
//
//			float4 hit = raycast(volume, make_uint2(x, y), view, nearPlane,
//					farPlane, step, largestep);
//			if (hit.w > 0) {
//				const float3 test = make_float3(hit);
//				const float3 surfNorm = volume.grad(test);
//				if (length(surfNorm) > 0) {
//					const float3 diff = normalize(light - test);
//					const float dir = fmaxf(dot(normalize(surfNorm), diff),
//							0.f);
//					const float3 col = clamp(make_float3(dir) + ambient, 0.f,
//							1.f) * 255;
//					out[pos] = make_uchar4(col.x, col.y, col.z, 0); // The forth value is a padding to align memory
//				} else {
//					out[pos] = make_uchar4(0, 0, 0, 0); // The forth value is a padding to align memory
//				}
//			} else {
//				out[pos] = make_uchar4(0, 0, 0, 0); // The forth value is a padding to align memory
//			}
//		}
//	}
	renderVolume_pencil(depthSize.x, depthSize.y, out, volume.size.x,
	                    volume.size.y, volume.size.z, volume.data,
	                    volume.dim, view, nearPlane, farPlane, step,
	                    largestep, light, ambient);
	TOCK("renderVolumeKernel", depthSize.x * depthSize.y);
}

#define OLDREDUCE 1
void reduceKernel(float * out, TrackData* J,
                  const uint2 Jsize, const uint2 size)
{
	TICK();
//	int blockIndex;
//#ifdef OLDREDUCE
//#pragma omp parallel for private (blockIndex)
//#endif
//	for (blockIndex = 0; blockIndex < 8; blockIndex++) {
//
//#ifdef OLDREDUCE
//		float S[112][32]; // this is for the final accumulation
//		// we have 112 threads in a blockdim
//		// and 8 blocks in a gridDim?
//		// ie it was launched as <<<8,112>>>
//		uint sline;// threadIndex.x
//		float sums[32];
//
//		for(int threadIndex = 0; threadIndex < 112; threadIndex++) {
//			sline = threadIndex;
//			float * jtj = sums+7;
//			float * info = sums+28;
//			for(uint i = 0; i < 32; ++i) sums[i] = 0;
//
//			for(uint y = blockIndex; y < size.y; y += 8 /*gridDim.x*/) {
//				for(uint x = sline; x < size.x; x += 112 /*blockDim.x*/) {
//					const TrackData & row = J[(x + y * Jsize.x)]; // ...
//
//					if(row.result < 1) {
//						// accesses S[threadIndex][28..31]
//						info[1] += row.result == -4 ? 1 : 0;
//						info[2] += row.result == -5 ? 1 : 0;
//						info[3] += row.result > -4 ? 1 : 0;
//						continue;
//					}
//					// Error part
//					sums[0] += row.error * row.error;
//
//					// JTe part
//					for(int i = 0; i < 6; ++i)
//					sums[i+1] += row.error * row.J[i];
//
//					// JTJ part, unfortunatly the double loop is not unrolled well...
//					jtj[0] += row.J[0] * row.J[0];
//					jtj[1] += row.J[0] * row.J[1];
//					jtj[2] += row.J[0] * row.J[2];
//					jtj[3] += row.J[0] * row.J[3];
//
//					jtj[4] += row.J[0] * row.J[4];
//					jtj[5] += row.J[0] * row.J[5];
//
//					jtj[6] += row.J[1] * row.J[1];
//					jtj[7] += row.J[1] * row.J[2];
//					jtj[8] += row.J[1] * row.J[3];
//					jtj[9] += row.J[1] * row.J[4];
//
//					jtj[10] += row.J[1] * row.J[5];
//
//					jtj[11] += row.J[2] * row.J[2];
//					jtj[12] += row.J[2] * row.J[3];
//					jtj[13] += row.J[2] * row.J[4];
//					jtj[14] += row.J[2] * row.J[5];
//
//					jtj[15] += row.J[3] * row.J[3];
//					jtj[16] += row.J[3] * row.J[4];
//					jtj[17] += row.J[3] * row.J[5];
//
//					jtj[18] += row.J[4] * row.J[4];
//					jtj[19] += row.J[4] * row.J[5];
//
//					jtj[20] += row.J[5] * row.J[5];
//
//					// extra info here
//					info[0] += 1;
//				}
//			}
//
//			for(int i = 0; i < 32; ++i) { // copy over to shared memory
//				S[sline][i] = sums[i];
//			}
//			// WE NO LONGER NEED TO DO THIS AS the threads execute sequentially inside a for loop
//
//		} // threads now execute as a for loop.
//		  //so the __syncthreads() is irrelevant
//
//		for(int ssline = 0; ssline < 32; ssline++) { // sum up columns and copy to global memory in the final 32 threads
//			for(unsigned i = 1; i < 112 /*blockDim.x*/; ++i) {
//				S[0][ssline] += S[i][ssline];
//			}
//			out[ssline+blockIndex*32] = S[0][ssline];
//		}
//#else
//		//new_reduce(blockIndex, out, J, Jsize, size);
//#endif
//
//	}
//
//	TooN::Matrix<8, 32, float, TooN::Reference::RowMajor> values(out);
//	for (int j = 1; j < 8; ++j) {
//		values[0] += values[j];
//		//std::cerr << "REDUCE ";for(int ii = 0; ii < 32;ii++)
//		//std::cerr << values[0][ii] << " ";
//		//std::cerr << "\n";
//	}

	reduce_pencil(out, Jsize.x, Jsize.y, J, size.x, size.y);
	TooN::Matrix<8, 32, float, TooN::Reference::RowMajor> values(out);
	for (int j = 1; j < 8; ++j)
		values[0] += values[j];
	TOCK("reduceKernel", 512);
}




bool Kfusion::preprocessing(const ushort * inputDepth, const uint2 inputSize)
{
	mm2metersKernel(floatDepth, computationSize, inputDepth, inputSize);
	bilateralFilterKernel(ScaledDepth[0], floatDepth, computationSize,
	                      gaussian, e_delta, radius);
	return true;
}

bool Kfusion::tracking(float4 k, float icp_threshold,
                       uint tracking_rate, uint frame)
{
	if (frame % tracking_rate != 0)
		return false;

	for (unsigned int i = 1; i < iterations.size(); ++i) {
		halfSampleRobustImageKernel(ScaledDepth[i], ScaledDepth[i - 1],
		                            make_uint2(computationSize.x / (int) pow(2, i),
		                                       computationSize.y / (int) pow(2, i)),
		                            e_delta * 3, 1);
	}

	uint2 localimagesize = computationSize;
	for (unsigned int i = 0; i < iterations.size(); ++i) {
		Matrix4 invK = getInverseCameraMatrix(k / float(1 << i));
		depth2vertexKernel(inputVertex[i], ScaledDepth[i], localimagesize, invK);
		vertex2normalKernel(inputNormal[i], inputVertex[i], localimagesize);
		localimagesize = make_uint2(localimagesize.x / 2, localimagesize.y / 2);
	}

	oldPose = pose;
	const Matrix4 projectReference = getCameraMatrix(k) * inverse(raycastPose);

	for (int level = iterations.size() - 1; level >= 0; --level) {
		uint2 localimagesize = make_uint2(computationSize.x / (int) pow(2, level),
		                                  computationSize.y / (int) pow(2, level));
		for (int i = 0; i < iterations[level]; ++i) {
			trackKernel(trackingResult, inputVertex[level], inputNormal[level],
			            localimagesize, vertex, normal, computationSize, pose,
			            projectReference, dist_threshold, normal_threshold);

			reduceKernel(reductionoutput, trackingResult,
			             computationSize, localimagesize);

			if (updatePoseKernel(pose, reductionoutput, icp_threshold))
				break;

		}
	}
	return checkPoseKernel(pose, oldPose, reductionoutput,
	                       computationSize, track_threshold);
}

bool Kfusion::raycasting(float4 k, float mu, uint frame)
{
	bool doRaycast = false;

	if (frame > 2) {
		raycastPose = pose;
		raycastKernel(vertex, normal, computationSize, volume,
		              raycastPose * getInverseCameraMatrix(k),
		              nearPlane, farPlane, step, 0.75f * mu);
	}

	return doRaycast;
}

bool Kfusion::integration(float4 k, uint integration_rate, float mu, uint frame)
{
	bool doIntegrate = checkPoseKernel(pose, oldPose, reductionoutput,
	                                   computationSize, track_threshold);

	if ((doIntegrate && ((frame % integration_rate) == 0)) || (frame <= 3)) {
		integrateKernel(volume, floatDepth, computationSize, inverse(pose),
		                getCameraMatrix(k), mu, maxweight);
		doIntegrate = true;
	} else {
		doIntegrate = false;
	}
	return doIntegrate;
}

void Kfusion::dumpVolume(std::string filename)
{
	std::ofstream fDumpFile;

	if (filename == "") {
		return;
	}

	std::cout << "Dumping the volumetric representation on file: "
	          << filename << std::endl;
	fDumpFile.open(filename.c_str(), std::ios::out | std::ios::binary);
	if (fDumpFile == NULL) {
		std::cout << "Error opening file: " << filename << std::endl;
		exit(1);
	}

	for (unsigned int i = 0;
	     i < volume.size.x * volume.size.y * volume.size.z;
	     i++) {
		fDumpFile.write((char *) (volume.data + i), sizeof(short));
	}

	fDumpFile.close();
}

void Kfusion::renderVolume(uchar4 * out, const uint2 outputSize, int frame,
                           int raycast_rendering_rate, float4 k,
                           float largestep)
{
	if (frame % raycast_rendering_rate == 0)
		renderVolumeKernel(out, outputSize, volume,
		                   *(this->viewPose) * getInverseCameraMatrix(k), nearPlane,
		                   farPlane * 2.0f, step, largestep, light, ambient);
}

void Kfusion::renderTrack(uchar4 * out, const uint2 outputSize)
{
	renderTrackKernel(out, trackingResult, outputSize);
}

void Kfusion::renderDepth(uchar4 * out, uint2 outputSize)
{
	renderDepthKernel(out, floatDepth, outputSize, nearPlane, farPlane);
}

void synchroniseDevices()
{
	// Nothing to do in the C++ implementation
}
