#ifndef PGM_UTILS_H
#define PGM_UTILS_H

#include "pch.h"
#include "AABB2D.h"


bool isInRange(double value, double lower, double upper);

Vector2 toVec2(Vector3 vec);

struct Hit {
    double alpha{};
    double beta{};
    double gamma{};

    Face face{};

    bool isvalid() {
        return (isInRange(alpha, 0.0, 1.0) && isInRange(beta, 0.0, 1.0) &&
                isInRange(gamma, 0.0, 1.0));
    }

    Vector3 getPoint(const VertexPositionGeometry& geom, const std::unordered_map<size_t, size_t>& matching ) {
        return alpha * geom.vertexPositions[matching.at(face.halfedge().vertex().getIndex())] +
               beta * geom.vertexPositions[matching.at(face.halfedge().next().vertex().getIndex())] +
               gamma * geom.vertexPositions[matching.at(face.halfedge().next().next().vertex().getIndex())];
    }

    Vector3 getNormal(const VertexPositionGeometry& geom, const std::unordered_map<size_t, size_t>& matching ) {
        Vector3 normal = alpha * geom.vertexNormals[matching.at(face.halfedge().vertex().getIndex())] +
                         beta * geom.vertexNormals[matching.at(face.halfedge().next().vertex().getIndex())] +
                         gamma * geom.vertexNormals[matching.at(face.halfedge().next().next().vertex().getIndex())];
        return normalize(normal);
    }
};

Hit checkPointTriangleIntersection(const VertexPositionGeometry& geom, const Face& face, const Vector2& point);
Hit checkPointMeshIntersection(const VertexPositionGeometry& geom, const Vector2& point);

std::vector<Vector3> sample(const VertexPositionGeometry& geom3D, const VertexPositionGeometry& geom2D,
                            const std::unordered_map<size_t, size_t>& matching, const Vec2Pair& box, int nSteps,
                            std::vector<Vector3>& normals);

void write_ply(const std::vector<Vector3>& data, const std::string& fn);
void getMatchingBC(std::unordered_map<size_t, size_t>& matchingBC, VertexPositionGeometry& geomB, VertexPositionGeometry& geomC);
void blendMeshes(VertexPositionGeometry& geomA, VertexPositionGeometry& geomB, VertexPositionGeometry& geomC,
                 std::unordered_map<size_t, size_t>& matchingBA, std::unordered_map<size_t, size_t>& matchingBC);
Vector3 pointTriangleDistance(const Vector3& point, const Face& f, const VertexPositionGeometry& geom);
Vector3 pointMeshClosestPoint(const Vector3& point, const VertexPositionGeometry& geom);


#endif //PGM_UTILS_H
