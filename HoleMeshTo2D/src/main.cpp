#include "pch.h"

#include "AABB2D.h"
#include "utils.h"


polyscope::SurfaceMesh *psMesh;
std::unordered_map<size_t, size_t> matching; // 2D -> 3D (B -> A)


void check(size_t vid, VertexPositionGeometry& geometry, std::vector<Vector3>& pts,
           std::unordered_map<size_t, size_t>& matching3D_2D, int& vertexNumber) {
    if (matching3D_2D.count(vid) == 0) {
        pts.push_back(geometry.vertexPositions[vid]);
        matching[vertexNumber] = vid;
        matching3D_2D[vid] = vertexNumber++;
    }
};

std::tuple<std::unique_ptr<ManifoldSurfaceMesh>, std::unique_ptr<VertexPositionGeometry>>
flattenHole(ManifoldSurfaceMesh& mesh, VertexPositionGeometry& geometry, int nTriangleStrips) {
    // Select faces
    int blId = 0;
    BoundaryLoop BL = mesh.boundaryLoop(blId);
    std::unordered_set<Face> selectedFaces;

    for (const Halfedge& he : BL.adjacentHalfedges())
        selectedFaces.insert(he.twin().face());


    for (int i = 0; i < nTriangleStrips; ++i) {
        std::unordered_set<Face> neighbours;
        for (const Face& face : selectedFaces) {
            for (const Face& neighbouringFace : face.adjacentFaces())
                neighbours.insert(neighbouringFace);
        }
        selectedFaces.insert(neighbours.begin(), neighbours.end());
    }

    for (const Face& f1 : mesh.faces()) { // TWIN
        int nSelected = 0;
        for (const Face& f2 : f1.adjacentFaces()) {
            if (selectedFaces.find(f2) != selectedFaces.end())
                nSelected++;
        }
        if (nSelected == 3)
            selectedFaces.insert(f1);
    }

    // Make mesh
    std::vector<Vector3> pts;
    std::vector<std::vector<size_t>> faces;

    matching.clear();
    std::unordered_map<size_t, size_t> matching3D_2D;
    int vertexNumber = 0;
    for (const Face& face : selectedFaces) {
        size_t v1Id = face.halfedge().vertex().getIndex();
        check(v1Id, geometry, pts, matching3D_2D, vertexNumber);
        size_t v2Id = face.halfedge().next().vertex().getIndex();
        check(v2Id, geometry, pts, matching3D_2D, vertexNumber);
        size_t v3Id = face.halfedge().next().next().vertex().getIndex();
        check(v3Id, geometry, pts, matching3D_2D, vertexNumber);
    }
    for (const Face& face : selectedFaces) {
        size_t v = matching3D_2D[face.halfedge().vertex().getIndex()];
        faces.push_back({v,
                         matching3D_2D[face.halfedge().next().vertex().getIndex()],
                         matching3D_2D[face.halfedge().next().next().vertex().getIndex()]});
    }

    std::unique_ptr<ManifoldSurfaceMesh> mesh2D;
    std::unique_ptr<VertexPositionGeometry> geometry2D;
    std::tie(mesh2D, geometry2D) = makeManifoldSurfaceMeshAndGeometry(faces, pts);

    Eigen::MatrixXd V(pts.size(), 3);
    Eigen::MatrixXi F(faces.size(), 3);
    Eigen::MatrixXd V_uv;

    for (size_t i = 0; i < pts.size(); ++i) {
        V(i, 0) = pts[i][0];
        V(i, 1) = pts[i][1];
        V(i, 2) = pts[i][2];
    }
    for (size_t i = 0; i < faces.size(); ++i) {
        F(i, 0) = faces[i][0];
        F(i, 1) = faces[i][1];
        F(i, 2) = faces[i][2];
    }

    // ARAP Param
    Eigen::MatrixXd initial_guess;
    Eigen::VectorXi bnd;
    igl::boundary_loop(F,bnd);
    Eigen::MatrixXd bnd_uv;
    igl::map_vertices_to_circle(V,bnd,bnd_uv);

    igl::harmonic(V,F,bnd,bnd_uv,1,initial_guess);

    igl::ARAPData arap_data;
    Eigen::MatrixXi b(2,1);
    b(0) = 0;
    b(1) = 1;
    double l = norm(pts[0] - pts[1]);
    Eigen::MatrixXd bc(2,2);
    bc << 0, 0, l, 0;

    // Initialize ARAP
    arap_data.max_iter = 100;
    arap_precomputation(V,F,2,b,arap_data);

    // Solve arap using the harmonic map as initial guess
    V_uv = initial_guess;
    arap_solve(bc,arap_data,V_uv);

    Eigen::MatrixXd V_uvdef(V.rows(),3);
    for (Eigen::Index i = 0; i < V_uv.rows(); ++i) {
        V_uvdef.coeffRef(i, 0) = V_uv.coeffRef(i,0);
        V_uvdef.coeffRef(i, 1) = 0;
        V_uvdef.coeffRef(i, 2) = V_uv.coeffRef(i,1);
    }

    for (Eigen::Index i = 0; i < V_uv.rows(); ++i) {
        pts[i] = Vector3{V_uv.coeffRef(i,0), 0, V_uv.coeffRef(i,1)};
    }

    return makeManifoldSurfaceMeshAndGeometry(faces, pts);
}


int main(int argc, char** argv) {

    // Parse filename
    if (argc != 3) {
        std::cerr << "2 arguments with path to .obj file and number of triangle strips are required." << std::endl;
        return EXIT_FAILURE;
    }
    std::string filename = argv[1];
    fs::path filepath(filename);
    std::cout << filename << std::endl;

    int nTriangleStrips = std::stoi(argv[2]);

    // Read obj
    std::unique_ptr<ManifoldSurfaceMesh> mesh;
    std::unique_ptr<VertexPositionGeometry> geometry;
    std::tie(mesh, geometry) = readManifoldSurfaceMesh(filename);


    polyscope::init();

    // Make hole 2D
    std::unique_ptr<ManifoldSurfaceMesh> mesh2D;
    std::unique_ptr<VertexPositionGeometry> geometry2D;

    std::tie(mesh2D, geometry2D) = flattenHole(*mesh, *geometry, nTriangleStrips);

    psMesh = polyscope::registerSurfaceMesh(
            polyscope::guessNiceNameFromPath(filename),
            geometry->inputVertexPositions, mesh->getFaceVertexList(),
            polyscopePermutations(*mesh));

    polyscope::registerSurfaceMesh(polyscope::guessNiceNameFromPath(filename) + "2D",
            geometry2D->inputVertexPositions, mesh2D->getFaceVertexList(),
            polyscopePermutations(*mesh2D));


    AABB2D box(*geometry2D);
    box.optimize();
    box.normalize();
    box.resize(1.0);
    box.subBoxes = 1;

    polyscope::registerSurfaceMesh(
            polyscope::guessNiceNameFromPath(filename) + "2D",
            geometry2D->inputVertexPositions, mesh2D->getFaceVertexList(),
            polyscopePermutations(*mesh2D));

    for (int i = 0; i < box.subBoxes*box.subBoxes; ++i) {
        std::vector<Vector3> normals;
        std::vector<Vector3> param = sample(*geometry, *geometry2D, matching, box.subBox(i), 64, normals);
        polyscope::registerPointCloud("Param", param);

        write_ply(param, filepath.stem().string() + std::to_string(i) + ".ply");
    }

    polyscope::show();

    return EXIT_SUCCESS;
}