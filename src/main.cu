#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <map>
#include <math.h>
#include "xyzio.h"
#include "dcdio.h"
#include <curand.h>
#include <curand_kernel.h>
#include <float.h>
#include "vec_utils.h"
#include "timer.h"

#define R_ACT (0.1f)
#define E_ACT (5000.0f) //J/mol
#define DH (5000.0f) //J/mol
#define DT (2.4e-6) //nsec
#define LIMIT (100.0f) //nm
#define PAIRSCUTOFF (16.0f)
#define MAXPAIRS (1000)
#define HARMCUTOFF (1.0f)
#define HARMSCALE (5000.0f)

#define TYPENONE (0)
#define TYPEA (1)
#define TYPEB (2)
#define TYPEC (3)

#define BLOCK_SIZE (256)

template <typename T>
class hd_vector {
public:
    void resize(size_t len_) {
        h.resize(len_);
        d.resize(len_);
    }
    void h2d() { d = h; }
    void d2h() { h = d; }
    T* d_ptr() { return thrust::raw_pointer_cast(d.data()); }
    T* h_ptr() { return thrust::raw_pointer_cast(h.data()); }
    thrust::host_vector<T> h;
    thrust::device_vector<T> d;
};

typedef struct
{
    hd_vector<float3> c;
    hd_vector<float3> v;
    hd_vector<float> m;
    hd_vector<int> t;
    hd_vector<int> nearest;
    hd_vector<float> nearest_dist;
} Particles;

typedef struct
{
    float3 *c;
    float3 *v;
    float *m;
    int *t;
    int *nearest;
    float *nearest_dist;
    float3 *f;
    int *pairsCount;
    int *pairs;
} d_Particles;

void checkCUDAError(){
	cudaError_t error = cudaGetLastError();
	if(error != cudaSuccess)
    {
		printf("CUDA error: %s \n", cudaGetErrorString(error));
        exit(0);
	}
}

__global__ void find_nearest(d_Particles parts, int atom_count)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < atom_count)
    {
        parts.nearest_dist[i] = FLT_MAX;
        parts.nearest[i] = -1;
        float3 pi = parts.c[i];
        for(unsigned int p = 0; p < parts.pairsCount[i]; p++)
        {
            int j = parts.pairs[p*atom_count+i];
            if( (parts.t[i] != parts.t[j]) && (parts.t[j] != TYPENONE) && (i != j))
            {
//                if(parts.nearest[i] >= 0)
                {
                    float3 pj = parts.c[j];
                    float dist = getDistance(pi,pj,LIMIT);

                    if(parts.nearest_dist[i] > dist)
                    {
                        parts.nearest_dist[i] = dist;
                        parts.nearest[i] = j;
                    }
                }
            }
        }
    }
}

__global__ void pairlistGPU(float3* r, int *t, int N, int* d_pairsCount, int* d_pairs){
	int i, j;
	i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < N)
    {
		float3 ri = r[i];
		int pairsCount = 0;
		for(j = 0; j < N; j++)
        {
			if ( (j != i) && (t[j] != TYPENONE) )
            {
				float3 rj = r[j];
				if( (getDistance(ri, rj, LIMIT) < PAIRSCUTOFF) && pairsCount < MAXPAIRS)
                {
					d_pairs[pairsCount*N + i] = j;
					pairsCount ++;
				}
			}
		}
		d_pairsCount[i] = pairsCount;
	}
}

__global__ void perform_collisions(d_Particles parts, curandState *state, int atom_count)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < atom_count)
    {
//        if( (parts.nearest_dist[i] < R_ACT) && (parts.nearest[parts.nearest[i]] == i) )
        if(parts.nearest[i] >= 0)
        {
            if( (parts.nearest_dist[i] < R_ACT) && (parts.nearest[parts.nearest[i]] == i) && (parts.t[i] == TYPEA) && (parts.t[parts.nearest[i]] == TYPEB) && (parts.t[i] != TYPENONE) )
            {
                int j = parts.nearest[i];
                float3 vi = parts.v[i];
                float ei = parts.m[i]*(vi.x*vi.x+vi.y*vi.y+vi.z*vi.z);
                float3 vj = parts.v[j];
                float ej = parts.m[j]*(vj.x*vj.x+vj.y*vj.y+vj.z*vj.z);
                if(ei + ej > E_ACT)
                {
                        float scale = sqrt(2*(ei+ej-E_ACT+DH)/(parts.m[i]+parts.m[j]));
                        float phi = 2*M_PI*(float)curand_uniform(&state[i]);
                        float teta = M_PI*((float)curand_uniform(&state[i]) - 0.5);
                        parts.v[j].x = sinf(teta)*cosf(phi)*scale;
                        parts.v[j].y = sinf(teta)*sin(phi)*scale;
                        parts.v[j].z = cosf(teta)*scale;
                        parts.t[j] = 3;
                        //destroy A
                        parts.c[i].x = 0;
                        parts.c[i].y = 0;
                        parts.c[i].z = 0;
                        parts.v[i].x = 0;
                        parts.v[i].y = 0;
                        parts.v[i].z = 0;
                        parts.t[i] = 0;

                }
            }
        }
    }
}

__global__ void integrate(d_Particles parts, int atom_count)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < atom_count)
    {
        if(parts.t[i] != TYPENONE)
        {
            parts.c[i].x += parts.v[i].x*DT;
            parts.c[i].y += parts.v[i].y*DT;
            parts.c[i].z += parts.v[i].z*DT;
            parts.c[i] = transferPBC(parts.c[i], LIMIT);
            parts.v[i].x += parts.f[i].x*DT/parts.m[i];
            parts.v[i].y += parts.f[i].y*DT/parts.m[i];
            parts.v[i].z += parts.f[i].z*DT/parts.m[i];
            parts.f[i].x = 0;
            parts.f[i].y = 0;
            parts.f[i].z = 0;
            
            if( (!isfinite(parts.c[i].x)) || (!isfinite(parts.c[i].y)) || (!isfinite(parts.c[i].z)))
            {
                parts.t[i] = TYPENONE;
            }
        }
    }
}

__global__ void initCurand(curandState *state, unsigned long seed)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, idx, idx, &state[idx]);
}

__global__ void computeLJPairlist(float3* r, float3* f, int* type, int N, int* d_pairsCount, int* d_pairs)
{
	int i, j, p;
	i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < N)
    {
		float3 fi = f[i];
		float3 ri = r[i];
		for(p = 0; p < d_pairsCount[i]; p++)
        {
			j = d_pairs[p*N + i];
            if(type[j] != TYPENONE)
            {
                float3 rj = r[j];
                float Bij = 1.0f;
                float epsilon = 3e-2f;
                float sigma = 2e-1f;
                float3 rij = getVector(ri, rj, LIMIT);
                float rijmod2 = rij.x*rij.x + rij.y*rij.y + rij.z*rij.z;
                float srij2 = sigma*sigma/rijmod2;	
                float srij6 = srij2*srij2*srij2;
                float df = -6.0f*epsilon*srij6*Bij/rijmod2;

                if(!isfinite(df))
                {
                    df = 0.0f;
                }

                fi.x += df*rij.x;
                fi.y += df*rij.y;
                fi.z += df*rij.z;
            }
		}
		f[i] = fi;
        if(!isfinite(f[i].x))
            f[i].x = 0.0f;
        if(!isfinite(f[i].y))
            f[i].y = 0.0f;
        if(!isfinite(f[i].z))
            f[i].z = 0.0f;
	}
}

__global__ void computeHarmPairlist(float3* r, float3* f, int* type, int N, int* d_pairsCount, int* d_pairs)
{
	int i, j, p;
	i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < N)
    {
		float3 fi = f[i];
		float3 ri = r[i];
		for(p = 0; p < d_pairsCount[i]; p++)
        {
			j = d_pairs[p*N + i];
            if(type[j] != TYPENONE)
            {
                float3 rj = r[j];
                float3 rij = getVector(ri, rj, LIMIT);
                float rijmod2 = rij.x*rij.x + rij.y*rij.y + rij.z*rij.z;
                float rijmod = sqrtf(rijmod2);
                float df = 0.0f;
                if(rijmod <= HARMCUTOFF)
                {
                    df += HARMSCALE*(rijmod - HARMCUTOFF)*(rijmod - HARMCUTOFF);
                }

                if(!isfinite(df))
                {
                    df = 0.0f;
                }

                fi.x += df*rij.x;
                fi.y += df*rij.y;
                fi.z += df*rij.z;
            }
		}
		f[i] = fi;
        if(!isfinite(f[i].x))
            f[i].x = 0.0f;
        if(!isfinite(f[i].y))
            f[i].y = 0.0f;
        if(!isfinite(f[i].z))
            f[i].z = 0.0f;
	}
}

int main(int argc, char **argv)
{
    cudaSetDevice(1);
    std::map<char, int> name2ind;
    name2ind['H'] = TYPEA;
    name2ind['O'] = TYPEB;
    name2ind['N'] = TYPEC;
    std::map<int, char> ind2name;
    ind2name[TYPEA] = 'H';
    ind2name[TYPEB] = 'O';
    ind2name[TYPEC] = 'N';
    std::map<char, float> name2mass;
    name2mass['H'] = 0.012;
    name2mass['O'] = 0.016;
    name2mass['N'] = 0.028;
    std::map<int, float> ind2mass;
    ind2mass[TYPEA] = 0.012;
    ind2mass[TYPEB] = 0.016;
    ind2mass[TYPEC] = 0.028;
    XYZ in_xyz;
    readXYZ(argv[1], &in_xyz);
    Particles part;
    std::cout<<"Total particles "<<in_xyz.atomCount<<std::endl;
    part.c.resize(in_xyz.atomCount);
    part.v.resize(in_xyz.atomCount);
    part.m.resize(in_xyz.atomCount);
    part.t.resize(in_xyz.atomCount);
    part.nearest.resize(in_xyz.atomCount);
    part.nearest_dist.resize(in_xyz.atomCount);
    for(unsigned int i = 0; i < in_xyz.atomCount; i++)
    {
        part.t.h[i] = name2ind[in_xyz.atoms[i].name];
        part.m.h[i] = name2mass[in_xyz.atoms[i].name];
        part.c.h[i].x = in_xyz.atoms[i].x;
        part.c.h[i].y = in_xyz.atoms[i].y;
        part.c.h[i].z = in_xyz.atoms[i].z;
        part.nearest.h[i] = -1;
    }
    readXYZ(argv[2], &in_xyz);
    for(unsigned int i = 0; i < in_xyz.atomCount; i++)
    {
        part.v.h[i].x = in_xyz.atoms[i].x;
        part.v.h[i].y = in_xyz.atoms[i].y;
        part.v.h[i].z = in_xyz.atoms[i].z;
    }
    part.c.h2d();
    part.v.h2d();
    part.m.h2d();
    part.t.h2d();
    part.nearest.h2d();
    part.nearest_dist.h2d();


    d_Particles d_part;
    d_part.c = part.c.d_ptr();
    d_part.v = part.v.d_ptr();
    d_part.m = part.m.d_ptr();
    d_part.t = part.t.d_ptr();
    d_part.nearest = part.nearest.d_ptr();
    d_part.nearest_dist = part.nearest_dist.d_ptr();
    cudaMalloc((void**)&(d_part.f), in_xyz.atomCount * sizeof(float3));
    cudaMalloc((void**)&(d_part.pairsCount), in_xyz.atomCount * sizeof(int));
    cudaMalloc((void**)&(d_part.pairs), in_xyz.atomCount* MAXPAIRS * sizeof(int));
    checkCUDAError();
    
    
    curandState *devState;
    cudaMalloc((void**)&devState, in_xyz.atomCount * sizeof(curandState));
    initCurand<<<(in_xyz.atomCount)/BLOCK_SIZE+1, BLOCK_SIZE>>>(devState, 100500);
    cudaDeviceSynchronize();

    DCD dcd;
    DCD vels;
    int Nsteps = 25000000;
    int pairsfreq = 100;
    int dcdfreq = 1000;
    createDCD(&dcd, 3*in_xyz.atomCount, Nsteps/dcdfreq, 0, 1, 100, 0,0,0,0);
    createDCD(&vels, 3*in_xyz.atomCount, Nsteps/dcdfreq, 0, 1, 100, 0,0,0,0);
    dcdOpenWrite(&dcd, "out.dcd");
    dcdOpenWrite(&vels, "out_vels.dcd");
    dcdWriteHeader(dcd);
    dcdWriteHeader(vels);
    std::cout<<"Starting simulation with "<<part.c.d.size()<<" particles."<<std::endl;
    initTimer();
    for(unsigned int step = 0; step < Nsteps; step++)
    {
        if(step%pairsfreq == 0)
        {
            pairlistGPU<<<(in_xyz.atomCount)/BLOCK_SIZE+1, BLOCK_SIZE>>>(d_part.c, d_part.t, in_xyz.atomCount, d_part.pairsCount, d_part.pairs);
        }
//        computeLJPairlist<<<(in_xyz.atomCount)/BLOCK_SIZE+1, BLOCK_SIZE>>>(d_part.c, d_part.f, d_part.t, in_xyz.atomCount, d_part.pairsCount, d_part.pairs);
        computeHarmPairlist<<<(in_xyz.atomCount)/BLOCK_SIZE+1, BLOCK_SIZE>>>(d_part.c, d_part.f, d_part.t, in_xyz.atomCount, d_part.pairsCount, d_part.pairs);
        find_nearest<<<(in_xyz.atomCount)/BLOCK_SIZE+1, BLOCK_SIZE>>>(d_part, in_xyz.atomCount);
        checkCUDAError();
//        part.nearest.d2h();
//        for(unsigned int i = 0; i < in_xyz.atomCount; i++)
//        {
//            std::cout<<i<<" "<<part.nearest.h[i]<<" "<<part.nearest.h[part.nearest.h[i]]<<std::endl;
//        }
//        exit(0);
        perform_collisions<<<(in_xyz.atomCount)/BLOCK_SIZE+1, BLOCK_SIZE>>>(d_part, devState, in_xyz.atomCount);
        checkCUDAError();
        integrate<<<(in_xyz.atomCount)/BLOCK_SIZE+1, BLOCK_SIZE>>>(d_part, in_xyz.atomCount);
        
        if(step%dcdfreq == 0)
        {
            for(unsigned int i = 0; i < 3*in_xyz.atomCount; i++)
            {
                dcd.frame.X[i] = 0;
                dcd.frame.Y[i] = 0;
                dcd.frame.Z[i] = 0;
                vels.frame.X[i] = 0;
                vels.frame.Y[i] = 0;
                vels.frame.Z[i] = 0;
            }
            int c_a = 0;
            int c_b = in_xyz.atomCount;
            int c_c = 2*in_xyz.atomCount;
            part.c.d2h();
            part.v.d2h();
            part.t.d2h();
            double E = 0;
            for(unsigned int i = 0; i < in_xyz.atomCount; i++)
            {   
                int offset;
                if(part.t.h[i] == TYPEA)
                {
                    offset = c_a;
                    c_a += 1;
                }
                if(part.t.h[i] == TYPEB)
                {
                    offset = c_b;
                    c_b += 1;
                }
                if(part.t.h[i] == TYPEC)
                {
                    offset = c_c;
                    c_c += 1;
                }
                dcd.frame.X[offset] = part.c.h[i].x;
                dcd.frame.Y[offset] = part.c.h[i].y;
                dcd.frame.Z[offset] = part.c.h[i].z;
                vels.frame.X[offset] = part.v.h[i].x;
                vels.frame.Y[offset] = part.v.h[i].y;
                vels.frame.Z[offset] = part.v.h[i].z;
                double v2 = 0;
                if (part.t.h[i] != TYPENONE)
                {
                    float m = ind2mass[part.t.h[i]];
                    v2 = vels.frame.X[offset]*vels.frame.X[offset];
                    v2 += vels.frame.Y[offset]*vels.frame.Y[offset];
                    v2 += vels.frame.Z[offset]*vels.frame.Z[offset];
                    E += m*v2/2;
                }
            }
            E = E/(c_a+c_b+c_c - 3*in_xyz.atomCount);
            std::cout<<step<<" "<<c_a<<" "<<c_b-in_xyz.atomCount<<" "<<c_c-in_xyz.atomCount*2<<" "<<2.0*E/3.0/8.314<<std::endl;
            dcdWriteFrame(dcd);
            dcdWriteFrame(vels);
            printTime(step);
        }
    }
    dcdClose(dcd);
    dcdClose(vels);
    return 0; 
}
