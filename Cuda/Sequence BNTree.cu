#define SEQUENCE_BNTREE
#ifdef SEQUENCE_BNTREE
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "timer.h"
#include "check.h"
#include <cuda_runtime.h>
#include <cfloat>

#define SOFTENING 1e-9f

#define GET_INFO int blockId = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;\
int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z) + (threadIdx.z * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;

#define MAXN 4096
#define epi 0.0001f
#define delta 0.000000000000001f

//#define DEBUG

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

struct TreeNode
{
    int bodyCnt;//internal:>1 leaf:1 nullptr:-1
    float3 minScale;
    float3 maxScale;
    union {
        float3 center;
        float3 pos;
    };
    float3 velocity;
    TreeNode* p[8];
}*root[20];

void GetBoundingBox(Body* bodies,int n,float3& minPos,float3& maxPos)
{
    while (n--)
    {
        Body body = bodies[n - 1];
        minPos.x = min(minPos.x, body.x);
        minPos.y = min(minPos.y, body.y);
        minPos.z = min(minPos.z, body.z);
        maxPos.x = max(maxPos.x, body.x);
        maxPos.y = max(maxPos.y, body.y);
        maxPos.z = max(maxPos.z, body.z);
    }
}

void Insert(TreeNode* node, Body body)
{
    if (node->bodyCnt == -1)
    {
        node->bodyCnt = 1;
        node->pos = make_float3(body.x, body.y, body.z);
        node->velocity = make_float3(body.vx, body.vy, body.vz);
        for (int i = 0; i < 8; i++)
        {
            node->p[i] = nullptr;
        }
    }
    else if (node->bodyCnt == 1)
    {
        float3 center = make_float3(0.5f * (node->minScale.x + node->maxScale.x), 0.5f * (node->minScale.y + node->maxScale.y), 0.5f * (node->minScale.z + node->maxScale.z));
        //+++
        if (node->pos.x > center.x && node->pos.y > center.y && node->pos.z > center.z)
        {
            node->p[0] = new TreeNode;
            node->p[0]->bodyCnt = -1;
            node->p[0]->minScale = center;
            node->p[0]->maxScale = node->maxScale;
            Insert(node->p[0], { node->pos.x,node->pos.y,node->pos.z,node->velocity.x,node->velocity.y,node->velocity.z });
        }
        //++-
        else if (node->pos.x > center.x && node->pos.y > center.y && node->pos.z < center.z)
        {
            node->p[1] = new TreeNode;
            node->p[1]->bodyCnt = -1;
            node->p[1]->minScale = make_float3(center.x, center.y, node->minScale.z);
            node->p[1]->maxScale = make_float3(node->maxScale.x, node->maxScale.y, center.z);
            Insert(node->p[1], { node->pos.x,node->pos.y,node->pos.z,node->velocity.x,node->velocity.y,node->velocity.z });
        }
        //+-+
        else if (node->pos.x > center.x && node->pos.y < center.y && node->pos.z > center.z)
        {
            node->p[2] = new TreeNode;
            node->p[2]->bodyCnt = -1;
            node->p[2]->minScale = make_float3(center.x, node->minScale.y, center.z);
            node->p[2]->maxScale = make_float3(node->maxScale.x, center.y, node->maxScale.z);
            Insert(node->p[2], { node->pos.x,node->pos.y,node->pos.z,node->velocity.x,node->velocity.y,node->velocity.z });
        }
        //+--
        else if (node->pos.x > center.x && node->pos.y < center.y && node->pos.z < center.z)
        {
            node->p[3] = new TreeNode;
            node->p[3]->bodyCnt = -1;
            node->p[3]->minScale = make_float3(center.x, node->minScale.y, node->minScale.z);
            node->p[3]->maxScale = make_float3(node->maxScale.x, center.y, center.z);
            Insert(node->p[3], { node->pos.x,node->pos.y,node->pos.z,node->velocity.x,node->velocity.y,node->velocity.z });
        }
        //-++
        else if (node->pos.x < center.x && node->pos.y > center.y && node->pos.z > center.z)
        {
            node->p[4] = new TreeNode;
            node->p[4]->bodyCnt = -1;
            node->p[4]->minScale = make_float3(node->minScale.x, center.y, center.z);
            node->p[4]->maxScale = make_float3(center.x, node->maxScale.y, node->maxScale.z);
            Insert(node->p[4], { node->pos.x,node->pos.y,node->pos.z,node->velocity.x,node->velocity.y,node->velocity.z });
        }
        //-+-
        else if (node->pos.x < center.x && node->pos.y > center.y && node->pos.z < center.z)
        {
            node->p[5] = new TreeNode;
            node->p[5]->bodyCnt = -1;
            node->p[5]->minScale = make_float3(node->minScale.x, center.y, node->minScale.z);
            node->p[5]->maxScale = make_float3(center.x, node->maxScale.y, center.z);
            Insert(node->p[5], { node->pos.x,node->pos.y,node->pos.z,node->velocity.x,node->velocity.y,node->velocity.z });
        }
        //--+
        else if (node->pos.x < center.x && node->pos.y < center.y && node->pos.z > center.z)
        {
            node->p[6] = new TreeNode;
            node->p[6]->bodyCnt = -1;
            node->p[6]->minScale = make_float3(node->minScale.x, node->minScale.y, center.z);
            node->p[6]->maxScale = make_float3(center.x, center.y, node->maxScale.z);
            Insert(node->p[6], { node->pos.x,node->pos.y,node->pos.z,node->velocity.x,node->velocity.y,node->velocity.z });
        }
        //---
        else if (node->pos.x < center.x && node->pos.y < center.y && node->pos.z < center.z)
        {
            node->p[7] = new TreeNode;
            node->p[7]->bodyCnt = -1;
            node->p[7]->minScale = node->minScale;
            node->p[7]->maxScale = center;
            Insert(node->p[7], { node->pos.x,node->pos.y,node->pos.z,node->velocity.x,node->velocity.y,node->velocity.z });
        }



        //+++
        if (body.x > center.x && body.y > center.y && body.z > center.z)
        {
            if (node->p[0] == nullptr)
            {
                node->p[0] = new TreeNode;
                node->p[0]->bodyCnt = -1;
                node->p[0]->minScale = center;
                node->p[0]->maxScale = node->maxScale;
            }
            Insert(node->p[0], body);
        }
        //++-
        else if (body.x > center.x && body.y > center.y && body.z < center.z)
        {
            if (node->p[1] == nullptr)
            {
                node->p[1] = new TreeNode;
                node->p[1]->bodyCnt = -1;
                node->p[1]->minScale = make_float3(center.x, center.y, node->minScale.z);
                node->p[1]->maxScale = make_float3(node->maxScale.x, node->maxScale.y, center.z);
            }
            Insert(node->p[1], body);
        }
        //+-+
        else if (body.x > center.x && body.y < center.y && body.z > center.z)
        {
            if (node->p[2] == nullptr)
            {
                node->p[2] = new TreeNode;
                node->p[2]->bodyCnt = -1;
                node->p[2]->minScale = make_float3(center.x, node->minScale.y, center.z);
                node->p[2]->maxScale = make_float3(node->maxScale.x, center.y, node->maxScale.z);
            }
            Insert(node->p[2], body);
        }
        //+--
        else if (body.x > center.x && body.y < center.y && body.z < center.z)
        {
            if (node->p[3] == nullptr)
            {
                node->p[3] = new TreeNode;
                node->p[3]->bodyCnt = -1;
                node->p[3]->minScale = make_float3(center.x, node->minScale.y, node->minScale.z);
                node->p[3]->maxScale = make_float3(node->maxScale.x, center.y, center.z);
            }
            Insert(node->p[3], body);
        }
        //-++
        else if (body.x < center.x && body.y > center.y && body.z > center.z)
        {
            if (node->p[4] == nullptr)
            {
                node->p[4] = new TreeNode;
                node->p[4]->bodyCnt = -1;
                node->p[4]->minScale = make_float3(node->minScale.x, center.y, center.z);
                node->p[4]->maxScale = make_float3(center.x, node->maxScale.y, node->maxScale.z);
            }
            Insert(node->p[4], body);
        }
        //-+-
        else if (body.x < center.x && body.y > center.y && body.z < center.z)
        {
            if (node->p[5] == nullptr)
            {
                node->p[5] = new TreeNode;
                node->p[5]->bodyCnt = -1;
                node->p[5]->minScale = make_float3(node->minScale.x, center.y, node->minScale.z);
                node->p[5]->maxScale = make_float3(center.x, node->maxScale.y, center.z);
            }
            Insert(node->p[5], body);
        }
        //--+
        else if (body.x < center.x && body.y < center.y && body.z > center.z)
        {
            if (node->p[6] == nullptr)
            {
                node->p[6] = new TreeNode;
                node->p[6]->bodyCnt = -1;
                node->p[6]->minScale = make_float3(node->minScale.x, node->minScale.y, center.z);
                node->p[6]->maxScale = make_float3(center.x, center.y, node->maxScale.z);
            }
            Insert(node->p[6], body);
        }
        //---
        else if (body.x < center.x && body.y < center.y && body.z < center.z)
        {
            if (node->p[7] == nullptr)
            {
                node->p[7] = new TreeNode;
                node->p[7]->bodyCnt = -1;
                node->p[7]->minScale = node->minScale;
                node->p[7]->maxScale = center;
            }
            Insert(node->p[7], body);
        }

        node->bodyCnt = 0;
        float3 sum{ .0f,.0f,.0f };
        for (int i = 0; i < 8; i++)
        {
            if (node->p[i] != nullptr)
            {
                sum.x += node->p[i]->bodyCnt * node->p[i]->center.x;
                sum.y += node->p[i]->bodyCnt * node->p[i]->center.y;
                sum.y += node->p[i]->bodyCnt * node->p[i]->center.y;
                node->bodyCnt += node->p[i]->bodyCnt;
            }
        }
        sum.x /= node->bodyCnt;
        sum.y /= node->bodyCnt;
        sum.z /= node->bodyCnt;
        if (node->bodyCnt == 0)
        {
            node->bodyCnt = 1;
        }
        node->center = sum;
        node->velocity = float3{ -1.f,-1.f,-1.f };
    }
    else if (node->bodyCnt > 1)
    {
        float3 center = make_float3(0.5f * (node->minScale.x + node->maxScale.x), 0.5f * (node->minScale.y + node->maxScale.y), 0.5f * (node->minScale.z + node->maxScale.z));
        //+++
        if (body.x > center.x && body.y > center.y && body.z > center.z)
        {
            if (node->p[0]==nullptr)
            {
                node->p[0] = new TreeNode;
                node->p[0]->bodyCnt = -1;
                node->p[0]->minScale = center;
                node->p[0]->maxScale = node->maxScale;
            }
            Insert(node->p[0], { body.x,body.y,body.z,body.vx,body.vy,body.vz });
        }
        //++-
        else if (body.x > center.x && body.y > center.y && body.z < center.z)
        {
            if (node->p[1] == nullptr)
            {
                node->p[1] = new TreeNode;
                node->p[1]->bodyCnt = -1;
                node->p[1]->minScale = make_float3(center.x, center.y, node->minScale.z);
                node->p[1]->maxScale = make_float3(node->maxScale.x, node->maxScale.y, center.z);
            }
            Insert(node->p[1], { body.x,body.y,body.z,body.vx,body.vy,body.vz });
        }
        //+-+
        else if (body.x > center.x && body.y < center.y && body.z > center.z)
        {
            if (node->p[2] == nullptr)
            {
                node->p[2] = new TreeNode;
                node->p[2]->bodyCnt = -1;
                node->p[2]->minScale = make_float3(center.x, node->minScale.y, center.z);
                node->p[2]->maxScale = make_float3(node->maxScale.x, center.y, node->maxScale.z);
            }
            Insert(node->p[2], { body.x,body.y,body.z,body.vx,body.vy,body.vz });
        }
        //+--
        else if (body.x > center.x && body.y < center.y && body.z < center.z)
        {
            if (node->p[3] == nullptr)
            {
                node->p[3] = new TreeNode;
                node->p[3]->bodyCnt = -1;
                node->p[3]->minScale = make_float3(center.x, node->minScale.y, node->minScale.z);
                node->p[3]->maxScale = make_float3(node->maxScale.x, center.y, center.z);
            }
            Insert(node->p[3], { body.x,body.y,body.z,body.vx,body.vy,body.vz });
        }
        //-++
        else if (body.x < center.x && body.y > center.y && body.z > center.z)
        {
            if (node->p[4] == nullptr)
            {
                node->p[4] = new TreeNode;
                node->p[4]->bodyCnt = -1;
                node->p[4]->minScale = make_float3(node->minScale.x, center.y, center.z);
                node->p[4]->maxScale = make_float3(center.x, node->maxScale.y, node->maxScale.z);
            }
            Insert(node->p[4], { body.x,body.y,body.z,body.vx,body.vy,body.vz });
        }
        //-+-
        else if (body.x < center.x && body.y > center.y && body.z < center.z)
        {
            if (node->p[5] == nullptr)
            {
                node->p[5] = new TreeNode;
                node->p[5]->bodyCnt = -1;
                node->p[5]->minScale = make_float3(node->minScale.x, center.y, node->minScale.z);
                node->p[5]->maxScale = make_float3(center.x, node->maxScale.y, center.z);
            }
            Insert(node->p[5], { body.x,body.y,body.z,body.vx,body.vy,body.vz });
        }
        //--+
        else if (body.x < center.x && body.y < center.y && body.z > center.z)
        {
            if (node->p[6] == nullptr)
            {
                node->p[6] = new TreeNode;
                node->p[6]->bodyCnt = -1;
                node->p[6]->minScale = make_float3(node->minScale.x, node->minScale.y, center.z);
                node->p[6]->maxScale = make_float3(center.x, center.y, node->maxScale.z);
            }
            Insert(node->p[6], { body.x,body.y,body.z,body.vx,body.vy,body.vz });
        }
        //---
        else if (body.x < center.x && body.y < center.y && body.z < center.z)
        {
            if (node->p[7] == nullptr)
            {
                node->p[7] = new TreeNode;
                node->p[7]->bodyCnt = -1;
                node->p[7]->minScale = node->minScale;
                node->p[7]->maxScale = center;
            }
            Insert(node->p[7], { body.x,body.y,body.z,body.vx,body.vy,body.vz });
        }

        node->bodyCnt = 0;
        float3 sum{ .0f,.0f,.0f };
        for (int i = 0; i < 8; i++)
        {
            if (node->p[i] != nullptr)
            {
                sum.x += node->p[i]->bodyCnt * node->p[i]->center.x;
                sum.y += node->p[i]->bodyCnt * node->p[i]->center.y;
                sum.y += node->p[i]->bodyCnt * node->p[i]->center.y;
                node->bodyCnt += node->p[i]->bodyCnt;
            }
        }
        sum.x /= node->bodyCnt;
        sum.y /= node->bodyCnt;
        sum.z /= node->bodyCnt;
        node->center = sum;
    }
}

void BuildTree(Body* bodies,int n, float3 minPos, float3 maxPos,int Iter)
{
    root[Iter] = new TreeNode;
    root[Iter]->maxScale = maxPos;
    root[Iter]->minScale = minPos;
    root[Iter]->bodyCnt = -1;
    for (int i = 0; i < 8; i++)
    {
        root[Iter]->p[i] = nullptr;
    }
    for (int i = 0; i < n; i++)
    {
        Insert(root[Iter], bodies[i]);
    }
}

float3 CalculateForce(float3 posa, float3 posb)
{
    float dx = posa.x - posb.x;
    float dy = posa.y - posb.y;
    float dz = posa.z - posb.z;
    float distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
    float invDist = rsqrtf(distSqr);
    float invDist3 = invDist * invDist * invDist;

    return float3{ dx * invDist3,dy * invDist3,dz * invDist3 };
}

float3 Calculate(TreeNode* node,Body body)
{
    float3 disVec = make_float3(node->center.x - body.x, node->center.y - body.y, node->center.z - body.z);
    float Sqrdis = disVec.x * disVec.x + disVec.y * disVec.y + disVec.z * disVec.z + SOFTENING;
    if (node->bodyCnt == 1)
    {
        if (Sqrdis < delta)
        {
            return float3{ .0f,.0f,.0f };
        }
        else
        {
            return CalculateForce(node->pos, float3{ body.x,body.y,body.z });
        }
    }
    else
    {
        float V2 = node->maxScale.x - node->minScale.x;
        /**/if (V2 * rsqrtf(Sqrdis) < epi)
        {
            return CalculateForce(node->center, float3{ body.x,body.y,body.z });
        }
        else
        {
            float3 sum{ .0f,.0f,.0f };
            for (int i = 0; i < 8; i++)
            {
                if (node->p[i] != nullptr)
                {
                    float3 d = Calculate(node->p[i], body);
                    sum.x += d.x;
                    sum.y += d.y;
                    sum.z += d.z;
                }
            }
            return sum;
        }
    }
}

void DeleteTree(TreeNode* node)
{
    for (int i = 0; i < 8; i++)
    {
        if (node->p[i] != nullptr)
        {
            DeleteTree(node->p[i]);
        }
    }
    delete node;
}

void randomizeBodies_Debug(float* data, int n) {
    Body* bodyBuf = (Body*)data;
    for (int i = 0; i < n; i++) {
        int id4 = i / 4;
        int id2 = i / 2;
        bodyBuf[i].x = 1 - 2 * (id4 & 1);
        bodyBuf[i].y = 1 - 2 * (id2 & 1);
        bodyBuf[i].z = 1 - 2 * (i & 1);
        bodyBuf[i].vx = .0f;
        bodyBuf[i].vy = .0f;
        bodyBuf[i].vz = .0f;
    }
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

    int bytes = nBodies * 6;
    float* buf;

    buf = new float[bytes];

    Body* bodybuf = (Body*)buf;

    /*
     * As a constraint of this exercise, `randomizeBodies` must remain a host function.
     */

#ifdef DEBUG
    randomizeBodies_Debug(buf, nBodies);
#else
    randomizeBodies(buf, 6 * nBodies); // Init pos / vel data
#endif

    double totalTime = 0.0;


    /*******************************************************************/
    // Do not modify these 2 lines of code.
    for (int iter = 0; iter < nIters; iter++) {
        StartTimer();
        /*******************************************************************/

        float3 minPos{ FLT_MAX,FLT_MAX,FLT_MAX };
        float3 maxPos{ FLT_MIN,FLT_MIN,FLT_MIN };

        GetBoundingBox(bodybuf, nBodies, minPos, maxPos);

        BuildTree(bodybuf, nBodies, minPos, maxPos, iter);

        for (int i = 0; i < nBodies; i++)
        {
            float3 F = Calculate(root[iter], bodybuf[i]);
            bodybuf[i].vx += dt * F.x;
            bodybuf[i].vy += dt * F.y;
            bodybuf[i].vz += dt * F.z;
        }

        for (int i = 0; i < nBodies; i++)
        {
            bodybuf[i].x += bodybuf[i].vx*dt;
            bodybuf[i].y += bodybuf[i].vy*dt;
            bodybuf[i].z += bodybuf[i].vz*dt;
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

    delete[] buf;
    for (int i = 0; i < nIters; i++)
    {
        DeleteTree(root[i]);
    }
}
#endif