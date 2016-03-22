#include <math.h>

#include "cuda_runtime.h"
#include "vector_types.h"

/*
 *  Some vector operators
 */
inline __host__ __device__ float3 operator+(float3 a, float3 b){
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

inline __host__ __device__ float3 operator-(float3 a, float3 b){
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

inline void __host__ __device__ operator-=(float3 &a, float3 b){
    a.x -= b.x; a.y -= b.y; a.z -= b.z;
}

inline void __host__ __device__ operator+=(float3 &a, float3 b){
    a.x += b.x; a.y += b.y; a.z += b.z;
}
inline __host__ __device__ float3 operator*(float a, float3 b){
    return make_float3(a*b.x, a*b.y, a*b.z);
}

inline __host__ __device__ float3 operator*(float3 a, float b){
    return make_float3(a.x*b, a.y*b, a.z*b);
}

inline void __host__ __device__ operator*=(float3 &a, float b){
    a.x *= b; a.y *= b; a.z *= b;
}

inline void __host__ __device__ operator/=(float3 &a, float b){
    a.x /= b; a.y /= b; a.z /= b;
}

inline __host__ __device__ float3 operator/(float3 a, float b){
    return make_float3(a.x/b, a.y/b, a.z/b);
}

/*
 *  Computes the length of the vector
 */
inline float __host__ __device__ len(float3 a){
    return sqrtf(a.x*a.x + a.y*a.y + a.z*a.z);
}

/*
 * Rounds up the components of the vector
 */
inline __host__ __device__ float3 rint(float3 a){
    return make_float3(rint(a.x), rint(a.y), rint(a.z));
}

/*
 * Returns the vector connecting points with coordinates r1 and r2
 * taking the periodic boundary into account
 */
inline __host__ __device__ float3 getVector(float3 ri, float3 rj, float L){
	float3 dr = rj -ri;
	dr -= rint(dr/L)*L;
	return dr;
}

/*
 * Returns the length of the vector connecting
 * points with coordinates r1 and r2 taking the
 * periodic boundary into account
 */
inline float __host__ __device__ getDistance(float3 ri, float3 rj, float L){
	float3 dr = getVector(ri, rj, L);
	return len(dr);
}

/*
 * Transfer the coordinates r through the boundary
 * of the cubical periodic box of size L*L*L. Assumes that
 * coordinates are not further then in the neighboring box.
 * The implementation is straightforward for better readability.
 */
inline float3 __host__ __device__ transferPBC(float3 r, float L){
	if(r.x > L){
		r.x -= 2*L;
	} else
	if(r.x < -L){
		r.x += 2*L;
	}
	if(r.y > L){
		r.y -= 2*L;
	} else
	if(r.y < -L){
		r.y += 2*L;
	}
	if(r.z > L){
		r.z -= 2*L;
	} else
	if(r.z < -L){
		r.z += 2*L;
	}
	return r;
}
