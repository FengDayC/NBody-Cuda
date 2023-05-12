//#define NBODY1
#ifdef NBODY1
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "timer.h"
#include "check.h"
#include <cuda_runtime.h>

#define SOFTENING 1e-9f

#define GET_INFO int blockId = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;\
int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z) + (threadIdx.z * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;

extern float rsqrtf(float f);

/*
 * Each body contains x, y, and z coordinate positions,
 * as well as velocities in the x, y, and z directions.
 */

typedef struct { float x, y, z, vx, vy, vz; } Body;

/*
 * Do not modify this function. A constraint of this exercise is
 * that it remain a host function.
 */

void randomizeBodies(float* data, int n) {
    for (int i = 0; i < n; i++) {
        data[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
    }
}

/*
 * This function calculates the gravitational impact of all bodies in the system
 * on all others, but does not update their positions.
 */

__device__
float3 bodyForce(Body* p, int n, int i)
{
    float3 F{ .0f, .0f, .0f };

    for (int j = 0; j < n; j++) {
        float dx = p[j].x - p[i].x;
        float dy = p[j].y - p[i].y;
        float dz = p[j].z - p[i].z;
        float distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
        float invDist = rsqrtf(distSqr);
        float invDist3 = invDist * invDist * invDist;

        F.x += dx * invDist3; F.y += dy * invDist3; F.z += dz * invDist3;
    }
    
    return F;
}

__global__
void calculateForce(float* buf, float dt, int nBody)
{
    GET_INFO

    Body * bodyBuf = (Body*)buf;

    float3 F = bodyForce(bodyBuf, nBody,threadId);

    Body body = bodyBuf[threadId];

    body.vx += dt * F.x;
    body.vy += dt * F.y;
    body.vz += dt * F.z;

    bodyBuf[threadId] = body;
}

__global__
void calculatePos(float* buf, float dt)
{
    GET_INFO

    Body* bodyBuf = (Body*)buf;
    Body body = bodyBuf[threadId];

    body.x += dt * body.vx;
    body.y += dt * body.vy;
    body.z += dt * body.vz;

    bodyBuf[threadId] = body;
}

int main(const int argc, const char** argv) {

    /*
     * Do not change the value for `nBodies` here. If you would like to modify it,
     * pass values into the command line.
     */

    int nBodies = 2 << 11;
    int salt = 0;
    if (argc > 1) nBodies = 2 << atoi(argv[1]);

    /*
     * This salt is for assessment reasons. Tampering with it will result in automatic failure.
     */

    if (argc > 2) salt = atoi(argv[2]);

    const float dt = 0.01f; // time step
    const int nIters = 10;  // simulation iterations

    int bytes = nBodies * sizeof(Body);
    float* buf;

    cudaMallocManaged(&buf, bytes);

    Body* p = (Body*)buf;

    /*
     * As a constraint of this exercise, `randomizeBodies` must remain a host function.
     */

    randomizeBodies(buf, 6 * nBodies); // Init pos / vel data

    double totalTime = 0.0;

    dim3 grid(4,1,1);
    dim3 block(16,16,4);


     /*******************************************************************/
     // Do not modify these 2 lines of code.
    for (int iter = 0; iter < nIters; iter++) {
        StartTimer();
    /*******************************************************************/

        /*
         * You will likely wish to refactor the work being done in `bodyForce`,
         * as well as the work to integrate the positions.
         */
        calculateForce <<<grid, block>>> (buf, dt, nBodies);
        
        cudaDeviceSynchronize();

        calculatePos <<<grid, block>>> (buf, dt);

        cudaDeviceSynchronize();

    /*******************************************************************/
    // Do not modify the code in this section.
        const double tElapsed = GetTimer() / 1000.0;
        totalTime += tElapsed;
    }
    
    double avgTime = totalTime / (double)(nIters);
    float billionsOfOpsPerSecond = 1e-9 * nBodies * nBodies / avgTime;

#ifdef ASSESS
    checkPerformance(buf, billionsOfOpsPerSecond, salt);
#else
    checkAccuracy(buf, nBodies);
    printf("%d Bodies: average %0.3f Billion Interactions / second\n", nBodies, billionsOfOpsPerSecond);
    salt += 1;
#endif
    /*******************************************************************/

    /*
     * Feel free to modify code below.
     */

    cudaFree(buf);
}
#endif