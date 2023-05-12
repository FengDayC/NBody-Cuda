#define NBODY
#ifdef NBODY2
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "timer.h"
#include "check.h"
#include <cuda_runtime.h>

#define SOFTENING 1e-9f

#define BLOCKCNT 32
#define THREADSPERBLOCK 128

#define GET_INFO int blockId = blockIdx.x;\
int threadBlockId = threadIdx.x;\
int threadTotalId = threadIdx.x+blockId*THREADSPERBLOCK;

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
__global__
void calculateForce(float* buf, float dt)
{
    GET_INFO

    Body* bodyBuf = (Body*)buf;

    int index = threadBlockId;

    float3 F{ .0f,.0f,.0f };

    Body myBody = bodyBuf[threadTotalId];

    __shared__ float3 pos[THREADSPERBLOCK];

    for (int i = 0; i < BLOCKCNT; i++)
    {
        Body body = bodyBuf[index];
        pos[threadBlockId] = float3{body.x,body.y,body.z};

        __syncthreads();

        for (int j = 0; j < THREADSPERBLOCK; j++) {
            float3 posj = pos[j];
            float dx = posj.x - myBody.x;
            float dy = posj.y - myBody.y;
            float dz = posj.z - myBody.z;
            float distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
            float invDist = rsqrtf(distSqr);
            float invDist3 = invDist * invDist * invDist;

            F.x += dx * invDist3;
            F.y += dy * invDist3;
            F.z += dz * invDist3;
        }

        __syncthreads();

        index += THREADSPERBLOCK;
    }

    bodyBuf[threadTotalId].vx += F.x * dt;
    bodyBuf[threadTotalId].vy += F.y * dt;
    bodyBuf[threadTotalId].vz += F.z * dt;
}

__global__
void calculatePos(float* buf, float dt)
{
    GET_INFO

    Body* bodyBuf = (Body*)buf;

    Body myBody = bodyBuf[threadTotalId];

    myBody.x += myBody.vx * dt;
    myBody.y += myBody.vy * dt;
    myBody.z += myBody.vz * dt;

    bodyBuf[threadTotalId] = myBody;
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
    cudaMallocHost(&buf, bytes);

    /*DEBUG*
    int* debugOutput;
    cudaMallocManaged(&debugOutput, bytes);
    /*DEBUG*/

    /*
     * As a constraint of this exercise, `randomizeBodies` must remain a host function.
     */

    randomizeBodies(buf, 6 * nBodies); // Init pos / vel data

    double totalTime = 0.0;

    float* device_buf;
    cudaMalloc(&device_buf, bytes);
    cudaMemcpy(device_buf, buf, bytes, cudaMemcpyKind::cudaMemcpyHostToDevice);

    /*******************************************************************/
    // Do not modify these 2 lines of code.
    for (int iter = 0; iter < nIters; iter++) {
        StartTimer();
        /*******************************************************************/

            /*
             * You will likely wish to refactor the work being done in `bodyForce`,
             * as well as the work to integrate the positions.
             */
        calculateForce << < {BLOCKCNT, 1, 1}, { THREADSPERBLOCK,1,1 } >> > (device_buf, dt);

        cudaDeviceSynchronize();

        calculatePos << < {BLOCKCNT, 1, 1}, { THREADSPERBLOCK,1,1 } >> > (device_buf, dt);

        cudaDeviceSynchronize();

        if ( iter + 1 == nIters)
        {
            cudaMemcpy(buf, device_buf, bytes, cudaMemcpyKind::cudaMemcpyDeviceToHost);
        }

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

    /*DEBUG OUTPUT
    if (buf != nullptr)
    {
        int max = -1;
        for (int i = 0; i < 64; i++)
        {
            printf("%.3f\n", buf[i]);
        }
    }
    /*DEBUG OUTPUT*/

    cudaFree(buf);
    cudaFree(device_buf);
}
#endif