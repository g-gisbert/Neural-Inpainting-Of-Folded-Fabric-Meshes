#ifndef PGM_AABB2D_H
#define PGM_AABB2D_H

#include "pch.h"

#define EPSILON 10e-5

struct Vec2Pair {
    Vector2 m_min;
    Vector2 m_max;
};

class AABB2D {
public:
    AABB2D() = default;

    AABB2D(VertexPositionGeometry& geom) : geometry(geom), subBoxes(1) {
        construct(0.0f);
    }

    void normalize();
    void resize(double factor);
    void construct(float theta);
    void optimize();
    Vec2Pair subBox(int i);
    Vector2 getRelativeCoord(float u, float v);

    Vector2 m_min;
    Vector2 m_max;
    VertexPositionGeometry& geometry;
    int subBoxes;
};



#endif //PGM_AABB2D_H
