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

  int preprocessing_pencil( const uint , const uint, const ushort * inputDepth,
			    const uint , const uint ,float * floatDepth, float * ScaledDepth,
			    int radius, float* gaussian, float e_delta);

  int tracking_pencil(unsigned int size0x, unsigned int size0y,
		      unsigned int size1x, unsigned int size1y,
		      unsigned int size2x, unsigned int size2y,
		      float * ScaledDepth0,
		      float * ScaledDepth1,
		      float * ScaledDepth2,
		      float3*,float3*,float3*,
		      float3*,float3*,float3*,
		      float3*,float3*,
		      TrackData*,
		      float*,
		      Matrix4*, Matrix4,
		      float,float,
		      Matrix4 k0,
		      Matrix4 k1,
		      Matrix4 k2,
		      int,int,int,
		      float e_delta) ;
    
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
	reductionoutput = (float*) prl_mem_get_host_mem(prl_mem_alloc(reductionoutput_size, prl_mem_host_nowrite));
	memset(reductionoutput, 0, reductionoutput_size);

	ScaledDepth = (float**)  malloc(sizeof(float*)  * iterations.size());
	inputVertex = (float3**) malloc(sizeof(float3*) * iterations.size());
	inputNormal = (float3**) malloc(sizeof(float3*) * iterations.size());

	for (unsigned int i = 0; i < iterations.size(); ++i) {
		size_t size = (computationSize.x * computationSize.y) / (int) pow(2, i);
		ScaledDepth[i] = (float*)  prl_mem_get_host_mem(prl_mem_alloc(sizeof(float)  * size, prl_mem_host_noaccess));
		memset(ScaledDepth[i], 0, sizeof(float) * size);

		inputVertex[i] = (float3*)  prl_mem_get_host_mem(prl_mem_alloc(sizeof(float3) * size, prl_mem_host_noaccess));
		memset(inputVertex[i], 0, sizeof(float3) * size);

		inputNormal[i] = (float3*) prl_mem_get_host_mem(prl_mem_alloc(sizeof(float3) * size, prl_mem_host_noaccess));
		memset(inputNormal[i], 0, sizeof(float3) * size);
	}

	size_t size = computationSize.x * computationSize.y;
	floatDepth     = (float*)     prl_mem_get_host_mem(prl_mem_alloc(sizeof(float)* size, prl_mem_host_noaccess));
	vertex         = (float3*)    prl_mem_get_host_mem(prl_mem_alloc(sizeof(float3)    * size, prl_mem_host_noaccess));
	normal         = (float3*)    prl_mem_get_host_mem(prl_mem_alloc(sizeof(float3)    * size, prl_mem_host_noaccess));
	trackingResult = (TrackData*) prl_mem_get_host_mem(prl_mem_alloc(sizeof(TrackData) * size, prl_mem_host_noaccess));

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
	size_t vol_size = volume.size.x * volume.size.y * volume.size.z * sizeof(short2);
	prl_mem vol_obj = prl_mem_manage_host(vol_size, volume.data, prl_mem_host_noaccess);
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

	bilateralFilter_pencil(size.x, size.y, out, in, size,
	                       (radius * 2 + 1), gaussian, e_d, r);
	TOCK("bilateralFilterKernel", size.x * size.y);
}

void depth2vertexKernel(float3 *vertex, const float *depth,
                        uint2 imageSize, const Matrix4 invK)
{
	TICK();
	depth2vertex_pencil(imageSize.x, imageSize.y, vertex, depth, invK);
	TOCK("depth2vertexKernel", imageSize.x * imageSize.y);
}

void vertex2normalKernel(float3 *out, const float3 *in, uint2 imageSize)
{
	TICK();
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

	mm2meters_pencil(outSize.x, outSize.y, out, inSize.x, inSize.y, in, ratio);
	TOCK("mm2metersKernel", outSize.x * outSize.y);
}

void halfSampleRobustImageKernel(float* out, const float* in, uint2 inSize,
                                 const float e_d, const int r)
{
	TICK();

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

	integrateKernel_pencil(vol.size.x, vol.size.y, vol.size.z, vol.dim,
	                       vol.data, depthSize.x, depthSize.y, depth,
	                       invTrack, K, mu, maxweight);
	TOCK("integrateKernel", vol.size.x * vol.size.y);
}


void raycastKernel(float3* vertex, float3* normal, uint2 inputSize,
                   const Volume integration, const Matrix4 view,
                   const float nearPlane, const float farPlane,
                   const float step, const float largestep)
{
	TICK();


	raycast_pencil(inputSize.x, inputSize.y, vertex, normal, integration.size.x,
	               integration.size.y, integration.size.z, integration.data,
	               integration.dim, view, nearPlane, farPlane, step, largestep);

	TOCK("raycastKernel", inputSize.x * inputSize.y);
}

bool updatePoseKernelX(Matrix4 & pose, const float * output, float icp_threshold)
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

inline Matrix4 Mat4TimeMat4( Matrix4 m1 , Matrix4 m2) {
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
return m3;
}


bool updatePoseKernel_OCL( Matrix4 * pose,   float * output, float icp_threshold) {


  for (int j = 1; j < 8; ++j) {
    for (int x = 0; x < 32; ++x) {
      output [x] += output [x + j * 32 ];
    }
  }


  // CONVERT A 8x32 Matrix to Matrix 6x6 and Vector 6

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
      bool bConverged_Or_Error = false;
      do
	{
	  bool result = false;
	  //  bConverged_Or_Error = Diagonalize_SubLoop(k, z);

	  const int k1 = k-1;
	  // 520 is here!
	  for(int l=k; l>=0; --l)
	    { // 530
	      const int l1 = l-1;
	      if((fabs(vOffDiagonal[l]) + anorm) == anorm)
		goto line_565;
	      if((fabs(vDiagonal[l1]) + anorm) == anorm) {		  
		line_540 ( l ,  k,  l1,  anorm,  vOffDiagonal,  vDiagonal,  mU) ;
		goto line_565;
	      }
	      continue;


	    line_565:
	      {
		// Check for convergence..
		z = vDiagonal[k];
		if(l == k) {
		  result = true; // convergence.
		  goto line_end_of_do;
		}
		if(nIterations == 30)
		  {
                    nError = k;
                    result = true; // convergence.
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
		result = false; // convergence.
		goto line_end_of_do;
		// EO IF NOT CONVERGED CHUNK
	      } // EO IF TESTS HOLD
	    } // 530
	  // Code should never get here!

	line_end_of_do :
	  bConverged_Or_Error = result;
	}
      while(!bConverged_Or_Error);

      if(nError) {
	return false;
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



    Matrix4 m1 = *pose;
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

    pose->data[0] = m3.data[0];
    pose->data[1] = m3.data[1];
    pose->data[2] = m3.data[2];
    pose->data[3] = m3.data[3];

    // Return validity test result of the tracking
    float xsqr = 0;
    for(int i=0; i<6; ++i) {
      xsqr += x[i] * x[i];
    }
    float lnorm =     sqrt(xsqr);

    return lnorm < icp_threshold;
    /* skipped test : lnorm < icp_threshold */

    

//   
//     Matrix4 toto;
//     toto.data[0] = pose->data[0];
//     toto.data[1] = pose->data[1];
//     toto.data[2] = pose->data[2];
//     toto.data[3] = pose->data[3];
//     toto = Mat4TimeMat4 (RR , toto);
//     pose->data[0] = toto.data[0];
//     pose->data[1] = toto.data[1];
//     pose->data[2] = toto.data[2];
//     pose->data[3] = toto.data[3];
//     // Return validity test result of the tracking
//     float xsqr = 0;
//     for(int i=0; i<6; ++i) {
//       xsqr += x[i] * x[i];
//     }
//     float lnorm =     sqrt(xsqr);
//   
//     /* skipped test : lnorm < icp_threshold */
//     return (lnorm < icp_threshold);
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

	renderNormal_pencil(normalSize.x, normalSize.y, out, normal);
	TOCK("renderNormalKernel", normalSize.x * normalSize.y);
}

void renderDepthKernel(uchar4* out, float * depth, uint2 depthSize,
                       const float nearPlane, const float farPlane)
{
	TICK();


	float rangeScale = 1 / (farPlane - nearPlane);

	renderDepth_pencil(depthSize.x, depthSize.y, out, depth, nearPlane, farPlane);
	TOCK("renderDepthKernel", depthSize.x * depthSize.y);
}

void renderTrackKernel(uchar4* out, const TrackData* data, uint2 outSize)
{
	TICK();

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

	reduce_pencil(out, Jsize.x, Jsize.y, J, size.x, size.y);
	TooN::Matrix<8, 32, float, TooN::Reference::RowMajor> values(out);
	for (int j = 1; j < 8; ++j)
		values[0] += values[j];
	TOCK("reduceKernel", 512);
}




bool Kfusion::preprocessing(const ushort * inputDepth, const uint2 inputSize)
{
  preprocessing_pencil( inputSize.x , inputSize.y,inputDepth, computationSize.x,  computationSize.y, floatDepth, ScaledDepth[0],  radius,  gaussian,  e_delta);

			     
  return true;
}

bool Kfusion::tracking(float4 k, float icp_threshold,
                       uint tracking_rate, uint frame)
{
	if (frame % tracking_rate != 0)
		return false;
	
	assert(iterations.size() == 3); // Bruno : assume 3 level of pyramid only (extremely reasonnable)
	
	oldPose = pose;
	const Matrix4 projectReference = getCameraMatrix(k) * inverse(raycastPose);

	tracking_pencil(computationSize.x/1, computationSize.y/1,
			computationSize.x/2, computationSize.y/2, 
			computationSize.x/4, computationSize.y/4,
			ScaledDepth[0], ScaledDepth[1], ScaledDepth[2],
			inputVertex[0], inputVertex[1], inputVertex[2],
			inputNormal[0], inputNormal[1], inputNormal[2],
			vertex,normal,
			trackingResult,
			reductionoutput,
			&pose, projectReference, dist_threshold, normal_threshold,
			getInverseCameraMatrix(k / float(1 << 0)),
			getInverseCameraMatrix(k / float(1 << 1)),
			getInverseCameraMatrix(k / float(1 << 2)),
			iterations[0],
			iterations[1],
			iterations[2],
			e_delta);
/*

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

*/	
	
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
	if (fDumpFile.fail()) {
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
