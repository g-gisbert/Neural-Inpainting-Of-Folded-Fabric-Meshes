#include "utils.h"


bool isInRange(double value, double lower, double upper) {
    if (value >= lower && value <= upper)
        return true;
    return false;
}

Vector2 toVec2(Vector3 vec) {
    return Vector2{vec.x, vec.z};
}

Hit checkPointTriangleIntersection(const VertexPositionGeometry& geom, const Face& face, const Vector2& point) {
    Vector2 AB = toVec2(geom.vertexPositions[face.halfedge().next().vertex()] - geom.vertexPositions[face.halfedge().vertex()]);
    Vector2 AC = toVec2(geom.vertexPositions[face.halfedge().next().next().vertex()] - geom.vertexPositions[face.halfedge().vertex()]);
    Vector2 AP = point - toVec2(geom.vertexPositions[face.halfedge().vertex()]);
    double d00 = dot(AB, AB);
    double d01 = dot(AB, AC);
    double d11 = dot(AC, AC);
    double d20 = dot(AP, AB);
    double d21 = dot(AP, AC);
    double denom = d00 * d11 - d01 * d01;
    double v = (d11 * d20 - d01 * d21) / denom;
    double w = (d00 * d21 - d01 * d20) / denom;
    double u = 1.0 - v - w;
    return Hit{u, v, w, face};
}

Hit checkPointMeshIntersection(const VertexPositionGeometry& geom, const Vector2& point) {

    for (const Face& face : geom.mesh.faces()) {
        Hit hit = checkPointTriangleIntersection(geom, face, point);
        if (hit.isvalid()) {
            return hit;
        }
    }
    return Hit{-1,-1,-1};
}

std::vector<Vector3> sample(const VertexPositionGeometry& geom3D, const VertexPositionGeometry& geom2D,
                            const std::unordered_map<size_t, size_t>& matching, const Vec2Pair& box, int nSteps,
                            std::vector<Vector3>& normals) {

    Vector2 start = box.m_min;
    Vector2 end = box.m_max;

    std::vector<Vector3> samples(nSteps*nSteps);
    std::vector<Vector3> debugPC(nSteps*nSteps);
    normals.assign(nSteps*nSteps, Vector3::zero());

    for (int i = 0; i < nSteps; ++i) {
        for (int j = 0; j < nSteps; ++j) {
            Vector2 current = Vector2{(1.0-static_cast<double>(j)/(static_cast<double>(nSteps)-1)) * start[0],
                                      (1.0-static_cast<double>(i)/(static_cast<double>(nSteps)-1)) * start[1]} +
                              Vector2{static_cast<double>(j)/(static_cast<double>(nSteps)-1) * end[0],
                                      static_cast<double>(i)/(static_cast<double>(nSteps)-1) * end[1]};
            Vector3 debug = Vector3{current.x, 0, current.y};
            debugPC[nSteps*i + j] = debug;
            if ( Hit hit = checkPointMeshIntersection(geom2D, current); hit.isvalid() ) {
                samples[nSteps*i + j] = hit.getPoint(geom3D, matching);
            } else {
                samples[nSteps*i + j] = Vector3{0.0, 0.0, 0.0};
            }
        }
    }

    polyscope::registerPointCloud("samples in 2D", debugPC);

    return samples;
}

void write_ply(const std::vector<Vector3>& data, const std::string& fn) {
    std::ofstream file { fn};

    for (const Vector3& elem : data) {
        file << elem[0] << " " << elem[1] << " " << elem[2] << "\n";
    }

    file.close();
}
