#include "AABB2D.h"

void AABB2D::normalize() {
    double diff = 0.5 * ((m_max.x - m_min.x) - (m_max.y - m_min.y));
    if (diff > 0) {
        m_min.y -= diff;
        m_max.y += diff;
    } else {
        m_min.x += diff;
        m_max.x -= diff;
    }
}

void AABB2D::resize(double factor) {
    Vector2 pad = 0.5 * (1.0 - factor) * (m_max - m_min);

    m_max -= pad;
    m_min += pad;
}

void AABB2D::construct(float theta) {
    const VertexData<Vector3>& vp = geometry.vertexPositions;
    Vector2 min = {vp[0][0], vp[0][2]};
    Vector2 max = {vp[0][0], vp[0][2]};

    Eigen::Matrix3f rot;
    rot << cosf(theta), 0.0f, sinf(theta), 0.0f, 1.0f, 0.0f, -sinf(theta), 0.0f, cosf(theta);

    for (const Vertex& v : geometry.mesh.vertices()) {
        Vector3 pos = vp[v];
        Eigen::Vector3f posEigen = Eigen::Vector3f{pos.x, pos.y, pos.z};
        posEigen = rot * posEigen;
        pos = Vector3{posEigen[0], posEigen[1], posEigen[2]};
        if (pos.x < min.x)
            min.x = pos.x;
        if (pos.z < min.y)
            min.y = pos.z;
        if (pos.x > max.x)
            max.x = pos.x;
        if (pos.z > max.y)
            max.y = pos.z;
    }
    m_min = min + EPSILON*Vector2{1, 1};
    m_max = max - EPSILON*Vector2{1, 1};
}

void AABB2D::optimize() {
    float theta = 0.0f;
    float bestTheta = 0.0f;
    float minLength = 1000000.0f;
    int N = 256;
    for (int i = 0; i < N; ++i) {
        theta = float(i) * 2.0f*M_PIf / float(N);
        construct(theta);
        normalize();
        if (m_max.x - m_min.x < minLength) {
            bestTheta = theta;
            minLength = m_max.x - m_min.x;
        }
    }

    for (const Vertex& v : geometry.mesh.vertices()) {
        Eigen::Matrix3f rot;
        rot << cosf(bestTheta), 0.0f, sinf(bestTheta), 0.0f, 1.0f, 0.0f, -sinf(bestTheta), 0.0f, cosf(bestTheta);
        Vector3 pos = geometry.vertexPositions[v];
        Eigen::Vector3f posEigen = Eigen::Vector3f{pos.x, pos.y, pos.z};
        posEigen = rot * posEigen;
        geometry.vertexPositions[v] = Vector3{posEigen[0], posEigen[1], posEigen[2]};
    }

    construct(0.0);
    normalize();
}

Vec2Pair AABB2D::subBox(int i) {
    Vec2Pair box;

    int x = i / subBoxes;
    int y = i - x*subBoxes;

    float delta = (m_max.x - m_min.x) / float(subBoxes + 1.0f);
    box.m_min = m_min + Vector2{x*delta, y*delta};
    box.m_max = m_min + Vector2{(x+2)*delta, (y+2)*delta};

    return box;
}

Vector2 AABB2D::getRelativeCoord(float u, float v) {
    float size = m_max.x - m_min.x;
    float relU = (u - m_min.x) / size * 64;
    float relV = (v - m_min.y) / size * 64;
    return Vector2{relU, relV};
}